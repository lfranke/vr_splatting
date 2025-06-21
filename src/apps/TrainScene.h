#pragma once
#include "models/Pipeline.h"
#include "src/lib/data/Dataset.h"
#include "src/lib/data/NeuralScene.h"
#ifdef TBLOGGER
#    include "tensorboard_logger.h"
#endif

typedef std::unique_ptr<
    torch::data::StatelessDataLoader<TorchSingleSceneDataset, torch::MultiDatasetSampler>,
    std::default_delete<torch::data::StatelessDataLoader<TorchSingleSceneDataset, torch::MultiDatasetSampler>>>
    dataloader_type_t;
class TrainScene
{
   public:
    TrainScene(std::vector<std::string> scene_dirs);

    SceneDataTrainSampler CroppedSampler(std::shared_ptr<SceneData> scene, std::vector<int> indices);

    SceneDataTrainSampler FullSampler(std::shared_ptr<SceneData> scene, std::vector<int> indices);

    //  typedef std::unique_ptr<torch::data::StatelessDataLoader<TorchSingleSceneDataset, torch::MultiDatasetSampler>>
    //      dataloader_t;

    dataloader_type_t DataLoader(std::vector<SceneDataTrainSampler>& train_cropped_samplers, bool train, int& n);

    void Load(torch::DeviceType device, int scene);

    void Unload(bool always_unload = false);

    void SetScene(int id);

    void Train(int epoch, bool v);

    void StartEpoch();

    int current_scene = -1;

    struct PerSceneData
    {
        std::shared_ptr<NeuralScene> scene;

        // for each camera one
        std::vector<torch::Tensor> eval_crop_mask;

        // all image indices that are not used during training.
        // -> we interpolate the metadata for them
        std::vector<int> not_training_indices;

        LossResult epoch_loss;
    };
    int original_pc_size = -1;

    std::vector<PerSceneData> data;
    std::vector<SceneDataTrainSampler> train_cropped_samplers, eval_samplers, test_samplers, test_cropped_samplers;
};

std::string EncodeImageToString(const Image& img, std::string type = "png");
#ifdef TBLOGGER
template <typename T>
inline void LogImage(TensorBoardLogger* tblogger, const TemplatedImage<T>& img, std::string name, int step)
{
    auto img_str = EncodeImageToString(img, "png");
    tblogger->add_image(name, step, img_str, img.h, img.w, channels(img.type));
}
#endif
torch::Tensor CropMask(int h, int w, int border);