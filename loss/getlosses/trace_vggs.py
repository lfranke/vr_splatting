import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from collections import OrderedDict
import os
from os.path import join
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# import lpips
example = torch.rand(1, 3, 224, 224)

##loss_fn_alex = lpips.LPIPS(net='alex')
##traced_script_module = torch.jit.trace(loss_fn_alex, (example, example))
##traced_script_module.save("traced_lpips.pt")
##
##loss_fn_vgg = lpips.LPIPS(net='vgg')
##traced_script_module = torch.jit.trace(loss_fn_vgg, (example, example))
##traced_script_module.save("traced_vgg_lpips.pt")
### frozen_optim_vgg_lpips = torch.jit.optimize_for_inference(torch.jit.script(loss_fn_vgg.eval()))
### frozen_optim_vgg_lpips.save("frozen_optim_vgg_lpips.pt")

if True:
    model = torchvision.models.vgg19(pretrained=True)


# torch.jit.save(torch.jit.script(model), 'vgg_script.pth')
# traced_script_module = torch.jit.trace(model, (example, example))
# traced_script_module.save("traced_vgg.pt")


######## https://github.com/alievk/npbg/blob/5bc6f8d18e61978f167f7dbb21787771fbd59bf6/npbg/models/conv.py#L15
###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):

        if mask_in is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


##https://github.com/alievk/npbg/blob/master/npbg/criterions/vgg_loss.py
class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(-1)


class VGGLoss(nn.Module):
    def __init__(self, net='caffe', partialconv=False, optimized=False, loss_func=F.l1_loss,
                 save_dir='.cache/torch/models'):
        super().__init__()

        self.loss_fn = loss_func
        self.partialconv = partialconv

        if net == 'pytorch':
            vgg19 = torchvision.models.vgg19(pretrained=True).features

            self.register_buffer('mean_', torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None])
            self.register_buffer('std_', torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None])

        elif net == 'pytorch16':
            vgg19 = torchvision.models.vgg16(pretrained=True).features

            self.register_buffer('mean_', torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None])
            self.register_buffer('std_', torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None])

        elif net == 'caffe':
            if not os.path.exists(join(save_dir, 'vgg_caffe_features.pth')):
                vgg_weights = torch.utils.model_zoo.load_url(
                    'https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth', model_dir=save_dir)

                map = {'classifier.6.weight': u'classifier.7.weight', 'classifier.6.bias': u'classifier.7.bias'}
                vgg_weights = OrderedDict([(map[k] if k in map else k, v) for k, v in vgg_weights.items()])

                model = torchvision.models.vgg19()
                model.classifier = nn.Sequential(View(), *model.classifier._modules.values())

                model.load_state_dict(vgg_weights)

                vgg19 = model.features
                os.makedirs(save_dir, exist_ok=True)
                torch.save(vgg19, join(save_dir, 'vgg_caffe_features.pth'))

                self.register_buffer('mean_',
                                     torch.FloatTensor([103.939, 116.779, 123.680])[None, :, None, None] / 255.)
                self.register_buffer('std_', torch.FloatTensor([1. / 255, 1. / 255, 1. / 255])[None, :, None, None])

            else:
                vgg19 = torch.load(join(save_dir, 'vgg_caffe_features.pth'))
                self.register_buffer('mean_',
                                     torch.FloatTensor([103.939, 116.779, 123.680])[None, :, None, None] / 255.)
                self.register_buffer('std_', torch.FloatTensor([1. / 255, 1. / 255, 1. / 255])[None, :, None, None])
        else:
            assert False

        vgg19_avg_pooling = []

        for weights in vgg19.parameters():
            weights.requires_grad = False

        for module in vgg19.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg19_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg19_avg_pooling.append(module)

        if optimized:
            self.layers = [3, 8, 17, 26, 35]
        else:
            self.layers = [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29]

        self.vgg19 = nn.Sequential(*vgg19_avg_pooling)

        # print(self.vgg19)

    def normalize_inputs(self, x):
        return (x - self.mean_) / self.std_

    def forward(self, input, target):
        loss = 0

        features_input = self.normalize_inputs(input)
        features_target = self.normalize_inputs(target)
        for i, layer in enumerate(self.vgg19):
            # if isinstance(layer, PartialConv2d):
            #    features_input  = layer(features_input, mask)
            #    features_target = layer(features_target, mask)
            # else:
            features_input = layer(features_input)
            features_target = layer(features_target)

            if i in self.layers:
                loss = loss + F.mse_loss(features_input,
                                         features_target)  # self.loss_fn(features_input, features_target)

        return loss


save_dir = '.cache/torch/models'

model = VGGLoss()

torch.jit.save(torch.jit.script(model), 'vgg_script_caffe.pth')
traced_script_module = torch.jit.trace(model, (example, example))
traced_script_module.save("traced_caffe_vgg.pt")
with torch.jit.optimized_execution(True):
    model = model.eval();
    traced_optimized_model = torch.jit.trace(model, (example, example))
    traced_optimized_model.save("traced_caffe_vgg_optim.pt")

model2 = VGGLoss('pytorch')

torch.jit.save(torch.jit.script(model2), 'vgg_script.pth')
traced_script_module = torch.jit.trace(model2, (example, example))
traced_script_module.save("traced_vgg.pt")

traced_script_module = torch.jit.trace(model2, (torch.rand(1, 3, 512, 512), torch.rand(1, 3, 512, 512)))
traced_script_module.save("traced_vgg_512.pt")

traced_script_module = torch.jit.trace(model2, (torch.rand(1, 3, 480, 480), torch.rand(1, 3, 480, 480)))
traced_script_module.save("traced_vgg_480.pt")

model4 = VGGLoss('pytorch', False, True)
traced_script_module = torch.jit.trace(model4, (torch.rand(1, 3, 512, 512), torch.rand(1, 3, 512, 512)))
traced_script_module.save("traced_vgg_512_optim.pt")

model5 = VGGLoss('pytorch', False, True, F.mse_loss)
traced_script_module = torch.jit.trace(model5, (torch.rand(1, 3, 512, 512), torch.rand(1, 3, 512, 512)))
traced_script_module.save("traced_vgg_512_optim_mse.pt")

model3 = VGGLoss('pytorch16')

torch.jit.save(torch.jit.script(model3), 'vgg16_script.pth')
traced_script_module = torch.jit.trace(model3, (example, example))
traced_script_module.save("traced_vgg16.pt")
