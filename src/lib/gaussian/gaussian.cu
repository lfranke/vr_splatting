// #include "debug_utils.cuh"
#include "gaussian.cuh"
// #include "read_utils.cuh"
#include <c10/cuda/CUDACachingAllocator.h>
#include <exception>
#include <models/MyAdam.h>
#include <thread>



GaussianModel::GaussianModel(int sh_degree) : _max_sh_degree(sh_degree) {}

torch::Tensor GaussianModel::Get_covariance(float scaling_modifier)
{
    //  if (inference_baked) return _cov_inference;
    auto L                 = build_scaling_rotation(scaling_modifier * Get_scaling(), _rotation);
    auto actual_covariance = torch::mm(L, L.transpose(1, 2));
    auto symm              = strip_symmetric(actual_covariance);
    return symm;
}

void GaussianModel::Bake_for_inference()
{
    _xyz.requires_grad_(false);
    _features_dc.requires_grad_(false);
    _features_rest.requires_grad_(false);
    _scaling.requires_grad_(false);
    _rotation.requires_grad_(false);
    _opacity.requires_grad_(false);
    //! first this, as Get_scaling is use in this function
    // auto L                 = build_scaling_rotation(1.f * Get_scaling(), _rotation);
    // auto actual_covariance = torch::mm(L, L.transpose(1, 2));

    inference_baked = true;
    _scaling        = torch::exp(_scaling).cuda().contiguous();

    _opacity            = torch::sigmoid(_opacity).cuda().contiguous();
    _rotation           = torch::nn::functional::normalize(_rotation).cuda().contiguous();
    _features_inference = torch::cat({_features_dc, _features_rest}, 1).cuda().contiguous();


    // _cov_inference = strip_symmetric(actual_covariance).cuda().contiguous();
}


/**
 * @brief Fetches the features of the Gaussian model
 *
 * This function concatenates _features_dc and _features_rest along the second dimension.
 *
 * @return Tensor of the concatenated features
 */
torch::Tensor GaussianModel::Get_features() const
{
    if (inference_baked) return _features_inference;
    auto features_dc   = _features_dc;
    auto features_rest = _features_rest;
    return torch::cat({features_dc, features_rest}, 1);
}

/**
 * @brief Increment the SH degree by 1
 *
 * This function increments the active_sh_degree by 1, up to a maximum of max_sh_degree.
 */
void GaussianModel::One_up_sh_degree()
{
    if (_active_sh_degree < _max_sh_degree)
    {
        _active_sh_degree++;
    }
}



/**
 * @brief Initialize Gaussian Model from a Point Cloud.
 *
 * This function creates a Gaussian model from a given PointCloud object. It also sets
 * the spatial learning rate scale. The model's features, scales, rotations, and opacities
 * are initialized based on the input point cloud.
 *
 * @param pcd The input point cloud
 * @param spatial_lr_scale The spatial learning rate scale
 */
void GaussianModel::Create_from_pcd(const Saiga::UnifiedMesh& pcd, float spatial_lr_scale)
{
    _spatial_lr_scale = spatial_lr_scale;

    std::vector<vec3> pcolors;
    std::vector<vec3> ppos;
    for (int i = 0; i < pcd.NumVertices(); ++i)
    {
        ppos.push_back(pcd.position[i]);
        pcolors.push_back(pcd.color[i].head<3>());
    }
    const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    _xyz                 = torch::from_blob(ppos.data(), {static_cast<long>(ppos.size()), 3}, pointType)
               .clone()
               .to(torch::kCUDA)
               .contiguous()
               .set_requires_grad(true);

    auto dist2 = torch::clamp_min(distCUDA2(_xyz.clone()), 0.0000001);
    dist2      = torch::clamp_max(dist2, 0.0001f);
    PrintTensorInfo(dist2);
    _scaling = torch::log(torch::sqrt(dist2))
                   .unsqueeze(-1)
                   .repeat({1, 3})
                   .to(torch::kCUDA)
                   .contiguous()
                   .set_requires_grad(true);
    _rotation = torch::zeros({_xyz.size(0), 4})
                    .index_put_({torch::indexing::Slice(), 0}, 1)
                    .to(torch::kCUDA)
                    .contiguous()
                    .set_requires_grad(true);
    _opacity =
        inverse_sigmoid(0.1 * torch::ones({_xyz.size(0), 1})).to(torch::kCUDA).contiguous().set_requires_grad(true);
    _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA).contiguous();

    // colors
    auto colorType   = torch::TensorOptions().dtype(torch::kFloat32);
    auto fused_color = RGB2SH(torch::from_blob(pcolors.data(), {static_cast<long>(pcolors.size()), 3}, colorType)
                                  .to(pointType)
                                  .clone()
                                  .contiguous() /
                              255.f)
                           .to(torch::kCUDA);

    // features
    auto features =
        torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))}).to(torch::kCUDA);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None),
                         torch::indexing::Slice(1, torch::indexing::None)},
                        0.0);
    _features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)})
                       .transpose(1, 2)
                       .contiguous()
                       .set_requires_grad(true);
    _features_rest = features
                         .index({torch::indexing::Slice(), torch::indexing::Slice(),
                                 torch::indexing::Slice(1, torch::indexing::None)})
                         .transpose(1, 2)
                         .contiguous()
                         .set_requires_grad(true);
}

/**
 * @brief Setup the Gaussian Model for training
 *
 * This function sets up the Gaussian model for training by initializing several
 * parameters and settings based on the provided OptimizationParameters object.
 *
 * @param params The OptimizationParameters object providing the settings for training
 */
void GaussianModel::Training_setup(const gs::param::OptimizationParameters& params)
{
    this->_optim_params           = params;
    this->_percent_dense          = params.percent_dense;
    this->_xyz_gradient_accum     = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);
    this->_xyz_gradient_accum_new = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);

    this->_denom              = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);
    this->_xyz_scheduler_args = Expon_lr_func(params.position_lr_init * this->_spatial_lr_scale,
                                              params.position_lr_final * this->_spatial_lr_scale,
                                              params.position_lr_delay_mult, params.position_lr_max_steps);

    CreateOptimizer(params);
}
void GaussianModel::CreateOptimizer(const gs::param::OptimizationParameters& params)
{
    std::vector<torch::optim::OptimizerParamGroup> optimizer_params_groups;
    optimizer_params_groups.reserve(6);
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup(
        {_xyz}, std::make_unique<torch::optim::AdamOptions>(params.position_lr_init * this->_spatial_lr_scale)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup(
        {_features_dc}, std::make_unique<torch::optim::AdamOptions>(params.feature_lr)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup(
        {_features_rest}, std::make_unique<torch::optim::AdamOptions>(params.feature_lr / 20.)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup(
        {_scaling}, std::make_unique<torch::optim::AdamOptions>(params.scaling_lr * this->_spatial_lr_scale)));
    optimizer_params_groups.push_back(torch::optim::OptimizerParamGroup(
        {_rotation}, std::make_unique<torch::optim::AdamOptions>(params.rotation_lr)));
    optimizer_params_groups.push_back(
        torch::optim::OptimizerParamGroup({_opacity}, std::make_unique<torch::optim::AdamOptions>(params.opacity_lr)));

    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[0].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[1].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[2].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[3].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[4].options()).eps(1e-15);
    static_cast<torch::optim::AdamOptions&>(optimizer_params_groups[5].options()).eps(1e-15);

    _optimizer =
        std::make_shared<torch::optim::Adam>(optimizer_params_groups, torch::optim::AdamOptions(0.f).eps(1e-15));
}

void GaussianModel::Update_learning_rate(float iteration)
{
    // This is hacky because you cant change in libtorch individual parameter learning rate
    // xyz is added first, since _optimizer->param_groups() return a vector, we assume that xyz stays first
    auto lr = _xyz_scheduler_args(iteration);
    static_cast<torch::optim::AdamOptions&>(_optimizer->param_groups()[0].options()).set_lr(lr);
}

void GaussianModel::Reset_opacity()
{
    // opacitiy activation
    auto new_opacity =
        inverse_sigmoid(torch::ones_like(_opacity, torch::TensorOptions().dtype(torch::kFloat32)) * 0.01f);

    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *_optimizer->state()[get_for_map(_optimizer->param_groups()[5].params()[0].unsafeGetTensorImpl())]));

    _optimizer->state().erase(get_for_map(_optimizer->param_groups()[5].params()[0].unsafeGetTensorImpl()));

    adamParamStates->exp_avg(torch::zeros_like(new_opacity));
    adamParamStates->exp_avg_sq(torch::zeros_like(new_opacity));
    // replace tensor
    _optimizer->param_groups()[5].params()[0] = new_opacity.set_requires_grad(true);
    _opacity                                  = _optimizer->param_groups()[5].params()[0];

    _optimizer->state()[get_for_map(_optimizer->param_groups()[5].params()[0].unsafeGetTensorImpl())] =
        std::move(adamParamStates);
}



void GaussianModel::Decay_opacity(float decay_value)
{
    // opacitiy activation
    auto new_opacity = inverse_sigmoid(Get_opacity() * decay_value);

    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *_optimizer->state()[get_for_map(_optimizer->param_groups()[5].params()[0].unsafeGetTensorImpl())]));

    _optimizer->state().erase(get_for_map(_optimizer->param_groups()[5].params()[0].unsafeGetTensorImpl()));

    adamParamStates->exp_avg(torch::zeros_like(new_opacity));
    adamParamStates->exp_avg_sq(torch::zeros_like(new_opacity));
    // replace tensor
    _optimizer->param_groups()[5].params()[0] = new_opacity.set_requires_grad(true);
    _opacity                                  = _optimizer->param_groups()[5].params()[0];

    _optimizer->state()[get_for_map(_optimizer->param_groups()[5].params()[0].unsafeGetTensorImpl())] =
        std::move(adamParamStates);
}
void prune_optimizer(torch::optim::Adam* optimizer, const torch::Tensor& mask, torch::Tensor& old_tensor,
                     int param_position)
{
    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *optimizer->state()[get_for_map(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl())]));
    optimizer->state().erase(get_for_map(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()));

    adamParamStates->exp_avg(adamParamStates->exp_avg().index_select(0, mask));
    adamParamStates->exp_avg_sq(adamParamStates->exp_avg_sq().index_select(0, mask));

    optimizer->param_groups()[param_position].params()[0] = old_tensor.index_select(0, mask).set_requires_grad(true);
    old_tensor = optimizer->param_groups()[param_position].params()[0];  // update old tensor
    optimizer->state()[get_for_map(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl())] =
        std::move(adamParamStates);
}

void printOptimizerState(torch::optim::Adam* optimizer, int param_position)
{
    if (param_position >= optimizer->param_groups().size()) return;
    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *optimizer->state()[get_for_map(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl())]));
    std::cout << "Param Pos " << param_position << ": "
              << TensorInfo(optimizer->param_groups()[param_position].params()[0]) << "; exp_avg "
              << TensorInfo(adamParamStates->exp_avg()) << "exp_avg_sq " << TensorInfo(adamParamStates->exp_avg_sq())
              << std::endl;
}

void GaussianModel::prune_points(torch::Tensor mask)
{
    // reverse to keep points
    auto valid_point_mask = ~mask;
    int true_count        = valid_point_mask.sum().item<int>();
    auto indices          = torch::nonzero(valid_point_mask == true)
                       .index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None),
                               torch::indexing::Slice(torch::indexing::None, 1)})
                       .squeeze(-1);
    prune_optimizer(_optimizer.get(), indices, _xyz, 0);
    prune_optimizer(_optimizer.get(), indices, _features_dc, 1);
    prune_optimizer(_optimizer.get(), indices, _features_rest, 2);
    prune_optimizer(_optimizer.get(), indices, _scaling, 3);
    prune_optimizer(_optimizer.get(), indices, _rotation, 4);
    prune_optimizer(_optimizer.get(), indices, _opacity, 5);

    _xyz_gradient_accum     = _xyz_gradient_accum.index_select(0, indices);
    _xyz_gradient_accum_new = _xyz_gradient_accum_new.index_select(0, indices);
    _denom                  = _denom.index_select(0, indices);
    _max_radii2D            = _max_radii2D.index_select(0, indices);
}

static void cat_tensors_to_optimizer(torch::optim::Adam* optimizer, torch::Tensor& extension_tensor,
                                     torch::Tensor& old_tensor, int param_position)
{
    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *optimizer->state()[get_for_map(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl())]));
    //  optimizer->state().erase((optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()));

    adamParamStates->exp_avg(torch::cat({adamParamStates->exp_avg(), torch::zeros_like(extension_tensor)}, 0));
    adamParamStates->exp_avg_sq(torch::cat({adamParamStates->exp_avg_sq(), torch::zeros_like(extension_tensor)}, 0));
    //   adamParamStates->max_exp_avg_sq(
    //       torch::cat({adamParamStates->max_exp_avg_sq(), torch::zeros_like(extension_tensor)}, 0));
    optimizer->param_groups()[param_position].params()[0] =
        torch::cat({old_tensor, extension_tensor}, 0).set_requires_grad(true).contiguous();
    old_tensor = optimizer->param_groups()[param_position].params()[0];

    optimizer->state()[get_for_map(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl())] =
        std::move(adamParamStates);
#if 0
    auto group       = optimizer->param_groups()[param_position];
    auto& p          = group.params().front();
    auto param_state = optimizer->state().find(p.unsafeGetTensorImpl());
    if (param_state == optimizer->state().end()) return;

    auto& adamParamStates = static_cast<torch::optim::AdamParamState&>(*optimizer->state()[p.unsafeGetTensorImpl()]);

    // auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
    //     *optimizer->state()[(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl())]));
    optimizer->state().erase((optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl()));

    adamParamStates.exp_avg(torch::cat({adamParamStates.exp_avg(), torch::zeros_like(extension_tensor)}, 0));
    adamParamStates.exp_avg_sq(torch::cat({adamParamStates.exp_avg_sq(), torch::zeros_like(extension_tensor)}, 0));

    // optimizer->param_groups()[param_position].params()[0] =
    //     torch::cat({old_tensor, extension_tensor.clone().contiguous()},
    //     0).clone().contiguous().set_requires_grad(true);
    // old_tensor = optimizer->param_groups()[param_position].params()[0];
    p.set_(torch::cat({old_tensor, extension_tensor.clone().contiguous()}, 0));
    old_tensor = (p);
    PrintTensorInfo(optimizer->param_groups()[param_position].params()[0]);
    // optimizer->state()[(optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl())] =
    //     std::move(adamParamStates);

    //   optimizer->state()[p.unsafeGetTensorImpl()] = std::move(adamParamStates);
#endif
}

void GaussianModel::densification_postfix(torch::Tensor& new_xyz, torch::Tensor& new_features_dc,
                                          torch::Tensor& new_features_rest, torch::Tensor& new_scaling,
                                          torch::Tensor& new_rotation, torch::Tensor& new_opacity)
{
    cat_tensors_to_optimizer(_optimizer.get(), new_xyz, _xyz, 0);
    cat_tensors_to_optimizer(_optimizer.get(), new_features_dc, _features_dc, 1);
    cat_tensors_to_optimizer(_optimizer.get(), new_features_rest, _features_rest, 2);
    cat_tensors_to_optimizer(_optimizer.get(), new_scaling, _scaling, 3);
    cat_tensors_to_optimizer(_optimizer.get(), new_rotation, _rotation, 4);
    cat_tensors_to_optimizer(_optimizer.get(), new_opacity, _opacity, 5);


    //_xyz           = torch::cat({_xyz, new_xyz}, 0).clone().contiguous().set_requires_grad(true);
    //_features_dc   = torch::cat({_features_dc, new_features_dc}, 0).clone().contiguous().set_requires_grad(true);
    //_features_rest = torch::cat({_features_rest, new_features_rest}, 0).clone().contiguous().set_requires_grad(true);
    //_scaling       = torch::cat({_scaling, new_scaling}, 0).clone().contiguous().set_requires_grad(true);
    //_rotation      = torch::cat({_rotation, new_rotation}, 0).clone().contiguous().set_requires_grad(true);
    //_opacity       = torch::cat({_opacity, new_opacity}, 0).clone().contiguous().set_requires_grad(true);
    //
    // std::cout << "pre_optim" << std::endl;
    // printModel();
    //
    // CreateOptimizer(_optim_params);

    _xyz_gradient_accum     = torch::zeros({_xyz.size(0), 1}).to(torch::kCUDA);
    _xyz_gradient_accum_new = torch::zeros({_xyz.size(0), 1}).to(torch::kCUDA);
    _denom                  = torch::zeros({_xyz.size(0), 1}).to(torch::kCUDA);
    _max_radii2D            = torch::zeros({_xyz.size(0)}).to(torch::kCUDA);
}

void GaussianModel::densify_and_split(torch::Tensor& grads, float grad_threshold, float scene_extent, float min_opacity,
                                      float max_screen_size)
{
    static const int N      = 2;
    const int n_init_points = _xyz.size(0);
    // Extract points that satisfy the gradient condition
    torch::Tensor padded_grad              = torch::zeros({n_init_points}).to(torch::kCUDA);
    padded_grad.slice(0, 0, grads.size(0)) = grads.squeeze();
    torch::Tensor selected_pts_mask =
        torch::where(padded_grad >= grad_threshold, torch::ones_like(padded_grad).to(torch::kBool),
                     torch::zeros_like(padded_grad).to(torch::kBool));
    selected_pts_mask =
        torch::logical_and(selected_pts_mask, std::get<0>(Get_scaling().max(1)) > _percent_dense * scene_extent);
    auto indices = torch::nonzero(selected_pts_mask.squeeze(-1) == true)
                       .index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None),
                               torch::indexing::Slice(torch::indexing::None, 1)})
                       .squeeze(-1);

    torch::Tensor stds    = Get_scaling().index_select(0, indices).repeat({N, 1});
    torch::Tensor means   = torch::zeros({stds.size(0), 3}).to(torch::kCUDA);
    torch::Tensor samples = torch::randn({stds.size(0), stds.size(1)}).to(torch::kCUDA) * stds + means;
    torch::Tensor rots    = build_rotation(_rotation.index_select(0, indices)).repeat({N, 1, 1});

    torch::Tensor new_xyz =
        torch::bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + _xyz.index_select(0, indices).repeat({N, 1}).clone();
    torch::Tensor new_scaling  = torch::log(Get_scaling().index_select(0, indices).repeat({N, 1}) / (0.8 * N)).clone();
    torch::Tensor new_rotation = _rotation.index_select(0, indices).repeat({N, 1}).clone();
    torch::Tensor new_features_dc   = _features_dc.index_select(0, indices).repeat({N, 1, 1}).clone();
    torch::Tensor new_features_rest = _features_rest.index_select(0, indices).repeat({N, 1, 1}).clone();
    torch::Tensor new_opacity       = _opacity.index_select(0, indices).repeat({N, 1}).clone();

    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity);

    torch::Tensor prune_filter =
        torch::cat({selected_pts_mask.squeeze(-1),
                    torch::zeros({N * selected_pts_mask.sum().item<int>()}).to(torch::kBool).to(torch::kCUDA)});
    // torch::Tensor prune_filter = torch::cat({selected_pts_mask.squeeze(-1), torch::zeros({N *
    // selected_pts_mask.sum().item<int>()})}).to(torch::kBool).to(torch::kCUDA);
    prune_filter = torch::logical_or(prune_filter, (Get_opacity() < min_opacity).squeeze(-1));
    prune_points(prune_filter);
    // auto prune_filter = (Get_opacity() < min_opacity).squeeze(-1);
    // prune_points(prune_filter);
}

void GaussianModel::densify_and_clone(torch::Tensor& grads, float grad_threshold, float scene_extent)
{
    // Extract points that satisfy the gradient condition
    torch::Tensor selected_pts_mask =
        torch::where(torch::linalg::vector_norm(grads, {2}, 1, true, torch::kFloat32) >= grad_threshold,
                     torch::ones_like(grads.index({torch::indexing::Slice()})).to(torch::kBool),
                     torch::zeros_like(grads.index({torch::indexing::Slice()})).to(torch::kBool))
            .to(torch::kLong);

    selected_pts_mask = torch::logical_and(
        selected_pts_mask, std::get<0>(Get_scaling().max(1)).unsqueeze(-1) <= _percent_dense * scene_extent);

    auto indices = torch::nonzero(selected_pts_mask.squeeze(-1) == true)
                       .index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None),
                               torch::indexing::Slice(torch::indexing::None, 1)})
                       .squeeze(-1);
    torch::Tensor new_xyz           = _xyz.index_select(0, indices);
    torch::Tensor new_features_dc   = _features_dc.index_select(0, indices);
    torch::Tensor new_features_rest = _features_rest.index_select(0, indices);
    torch::Tensor new_opacity       = _opacity.index_select(0, indices);
    torch::Tensor new_scaling       = _scaling.index_select(0, indices);
    torch::Tensor new_rotation      = _rotation.index_select(0, indices);

    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity);
}

void GaussianModel::Densify_and_prune(float max_grad, float min_opacity, float extent, float max_screen_size)
{
    //   printModel();
    torch::Tensor grads = _xyz_gradient_accum / _denom;
    grads.index_put_({grads.isnan()}, 0.0);

    torch::Tensor grads_new = _xyz_gradient_accum_new / _denom;
    grads_new.index_put_({grads.isnan()}, 0.0);

    densify_and_clone(grads, max_grad, extent);
    densify_and_split(grads_new, max_grad, extent, min_opacity, max_screen_size);
    printModel();
    c10::cuda::CUDACachingAllocator::emptyCache();
}


void GaussianModel::Add_densification_stats(torch::Tensor& viewspace_point_tensor, torch::Tensor& update_filter,
                                            bool abs_grad)
{
    if (abs_grad)
    {
        // newer densification metrics
        _xyz_gradient_accum_new.index_put_(
            {update_filter},
            _xyz_gradient_accum_new.index_select(0, update_filter.nonzero().squeeze()) +
                viewspace_point_tensor.grad().index_select(0, update_filter.nonzero().squeeze()).slice(1, 2, 3));
    }

    _xyz_gradient_accum.index_put_({update_filter},
                                   _xyz_gradient_accum.index_select(0, update_filter.nonzero().squeeze()) +
                                       viewspace_point_tensor.grad()
                                           .index_select(0, update_filter.nonzero().squeeze())
                                           .slice(1, 0, 2)
                                           .norm(2, -1, true));

    _denom.index_put_({update_filter}, _denom.index_select(0, update_filter.nonzero().squeeze()) + 1);
}

std::vector<std::string> GaussianModel::construct_list_of_attributes()
{
    std::vector<std::string> attributes = {"x", "y", "z", "nx", "ny", "nz"};

    for (int i = 0; i < _features_dc.size(1) * _features_dc.size(2); ++i)
        attributes.push_back("f_dc_" + std::to_string(i));

    for (int i = 0; i < _features_rest.size(1) * _features_rest.size(2); ++i)
        attributes.push_back("f_rest_" + std::to_string(i));

    attributes.emplace_back("opacity");

    for (int i = 0; i < _scaling.size(1); ++i) attributes.push_back("scale_" + std::to_string(i));

    for (int i = 0; i < _rotation.size(1); ++i) attributes.push_back("rot_" + std::to_string(i));

    return attributes;
}
void GaussianModel::printModel(bool include_optimizer)
{
    std::cout << "========" << std::endl;
    std::cout << "Gaussian Model: " << std::endl;
    std::cout << "\t _xyz: " << TensorInfo(_xyz) << std::endl;
    std::cout << "\t _features_dc: " << TensorInfo(_features_dc) << std::endl;
    std::cout << "\t _features_rest: " << TensorInfo(_features_rest) << std::endl;
    std::cout << "\t _scaling: " << TensorInfo(_scaling) << std::endl;
    std::cout << "\t _rotation: " << TensorInfo(_rotation) << std::endl;
    std::cout << "\t _opacity: " << TensorInfo(_opacity) << std::endl;
    if (include_optimizer)
    {
        std::cout << "\t _xyz_gradient_accum: " << TensorInfo(_xyz_gradient_accum) << std::endl;
        std::cout << "\t _xyz_gradient_accum_new: " << TensorInfo(_xyz_gradient_accum_new) << std::endl;
        std::cout << "\t _denom: " << TensorInfo(_denom) << std::endl;
        std::cout << "\t _max_radii2D: " << TensorInfo(_max_radii2D) << std::endl;

        std::cout << "Optimizer:" << std::endl;
        for (int i = 0; i < 6; ++i) printOptimizerState(_optimizer.get(), i);
    }
    std::cout << "========" << std::endl;
}

std::pair<vec3, vec3> GaussianModel::Get_aabb()
{
    using namespace torch::indexing;
    std::cout << _xyz.sizes() << std::endl;
    std::cout << TensorInfo(_xyz.index({Slice(), Slice(0, 1)})) << TensorInfo(_xyz.index({Slice(), Slice(1, 2)}))
              << TensorInfo(_xyz.index({Slice(), Slice(2, 3)})) << std::endl;
    std::cout << TensorInfo(_xyz) << std::endl;
    //_xyz.index({Slice()})
    vec3 min(_xyz.index({Slice(), Slice(0, 1)}).min().item<float>(),
             _xyz.index({Slice(), Slice(1, 2)}).min().item<float>(),
             _xyz.index({Slice(), Slice(2, 3)}).min().item<float>());
    vec3 max(_xyz.index({Slice(), Slice(0, 1)}).max().item<float>(),
             _xyz.index({Slice(), Slice(1, 2)}).max().item<float>(),
             _xyz.index({Slice(), Slice(2, 3)}).max().item<float>());
    return {min, max};
}

void GaussianModel::SaveCheckpoint(const std::string& checkpoint_prefix, bool with_optimizer)
{
    std::string gs_dir = checkpoint_prefix + "/gaussian_model/";
    std::filesystem::create_directories(gs_dir);
    torch::save(_xyz, gs_dir + "xyz.pth");
    torch::save(_features_dc, gs_dir + "features_dc.pth");
    torch::save(_features_rest, gs_dir + "features_rest.pth");
    torch::save(_scaling, gs_dir + "scaling.pth");
    torch::save(_rotation, gs_dir + "rotation.pth");
    torch::save(_opacity, gs_dir + "opacity.pth");


    if (with_optimizer)
    {
        torch::save(_max_radii2D, gs_dir + "max_radii2D.pth");
        torch::save(_xyz_gradient_accum, gs_dir + "xyz_gradient_accum.pth");
        torch::save(_xyz_gradient_accum_new, gs_dir + "_xyz_gradient_accum_new.pth");
        torch::save(_denom, gs_dir + "denom.pth");

        torch::save(*_optimizer, gs_dir + "_optimizer.pth");
    }
}
void GaussianModel::LoadCheckpoint(const std::string& checkpoint_dir, bool with_optimizer)
{
    std::string gs_dir = checkpoint_dir + "/gaussian_model/";
    if (!std::filesystem::exists(gs_dir))
    {
        std::cerr << "No GS Model" << std::endl;
        return;
    }
    torch::load(_xyz, gs_dir + "xyz.pth");
    torch::load(_features_dc, gs_dir + "features_dc.pth");
    torch::load(_features_rest, gs_dir + "features_rest.pth");
    torch::load(_scaling, gs_dir + "scaling.pth");
    torch::load(_rotation, gs_dir + "rotation.pth");
    torch::load(_opacity, gs_dir + "opacity.pth");
    if (with_optimizer)
    {
        torch::load(_xyz_gradient_accum, gs_dir + "xyz_gradient_accum.pth");
        torch::load(_xyz_gradient_accum_new, gs_dir + "xyz_gradient_accum.pth");
        torch::load(_max_radii2D, gs_dir + "max_radii2D.pth");
        torch::load(_denom, gs_dir + "denom.pth");
    }
}

void GaussianModel::LoadCheckpointForTraining(const std::string& checkpoint_dir,
                                              const gs::param::OptimizationParameters& params)
{
    std::string gs_dir = checkpoint_dir + "/gaussian_model/";

    if (!std::filesystem::exists(gs_dir))
    {
        std::cerr << "No GS Model" << std::endl;
        return;
    }
    LoadCheckpoint(checkpoint_dir);

    this->_optim_params = params;

    CreateOptimizer(params);
    torch::load(*_optimizer, gs_dir + "_optimizer.pth");
}

void GaussianModel::Save_ply(const std::filesystem::path& file_path, int iteration, bool isLastIteration)
{
    SAIGA_ASSERT("NOT IMPLEMENTED YET");
#if 0
    std::cout << "Saving at " << std::to_string(iteration) << " iterations\n";
    auto folder = file_path / ("point_cloud/iteration_" + std::to_string(iteration));
    std::filesystem::create_directories(folder);

    auto xyz       = _xyz.cpu().contiguous();
    auto normals   = torch::zeros_like(xyz);
    auto f_dc      = _features_dc.transpose(1, 2).flatten(1).cpu().contiguous();
    auto f_rest    = _features_rest.transpose(1, 2).flatten(1).cpu().contiguous();
    auto opacities = _opacity.cpu();
    auto scale     = _scaling.cpu();
    auto rotation  = _rotation.cpu();

    std::vector<torch::Tensor> tensor_attributes = {xyz.clone(),       normals.clone(), f_dc.clone(),    f_rest.clone(),
                                                    opacities.clone(), scale.clone(),   rotation.clone()};
    auto attributes                              = construct_list_of_attributes();
    std::thread t                                = std::thread([folder, tensor_attributes, attributes]()
                                { Write_output_ply(folder / "point_cloud.ply", tensor_attributes, attributes); });

    if (isLastIteration)
    {
        t.join();
    }
    else
    {
        t.detach();
    }
#endif
}