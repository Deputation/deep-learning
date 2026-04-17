#include "Arena.h"
#include "ArenaArrayOperations.h"
#include "ArenaTensorNode.h"
#include "ArenaWatcher.h"
#include "Functions/ActivationFunctions.h"
#include "Functions/LossFunctions.h"
#include "Layers/ActivationLayer.h"
#include "Layers/LinearLayer.h"
#include "Model.h"
#include "Optimizers/Adam.h"
#include "Optimizers/AdamWeightDecay.h"
#include "Schedulers/FixedRateScheduler.h"
#include "TensorOperations.h"
#include "TensorUtilities.h"

#include <Tensor.h>
#include <assert.h>
#include <fstream>
#include <iostream>

Tensor<unsigned char> ParseMnistLabels(const std::string &Path) {
  std::ifstream File(Path, std::ios::binary);

  if (!File.good()) {
    assert(false && "Images file could not be read.");
  }

  int Magic;
  File.read((char *)&Magic, sizeof(int));
  Magic = __builtin_bswap32(Magic);

  if (Magic != 2049) {
    assert(false && "Magic number could not be read.");
  }

  int DimensionalMask = 0x000000FF;
  int DataTypeMask = 0x0000FF00;

  [[maybe_unused]] auto Dimensions = Magic & DimensionalMask;
  auto DataTypeSize = (Magic & DataTypeMask) >> 8;

  if (DataTypeSize != 8) {
    assert(false && "Unexpected data type size.");
  }

  assert(Dimensions == 1);

  int Size;
  File.read((char *)&Size, sizeof(int));
  Size = __builtin_bswap32(Size);

  ArenaArray<size_t> TensorDimensions({static_cast<size_t>(Size), 1});

  Tensor<unsigned char> Result(TensorDimensions);
  for (size_t i = 0; i < Result.Data.Size; i++) {
    unsigned char ByteValue;
    File.read((char *)&ByteValue, sizeof(unsigned char));
    Result.Data[i] = ByteValue;
  }

  return Result;
}

Tensor<double> ParseMnistImages(const std::string &Path) {
  std::ifstream File(Path, std::ios::binary);

  if (!File.good()) {
    assert(false && "Images file could not be read.");
  }

  int Magic;
  File.read((char *)&Magic, sizeof(int));
  Magic = __builtin_bswap32(Magic);

  if (Magic != 2051) {
    assert(false && "Magic number could not be read.");
  }

  int DimensionalMask = 0x000000FF;
  int DataTypeMask = 0x0000FF00;

  auto Dimensions = Magic & DimensionalMask;
  auto DataTypeSize = (Magic & DataTypeMask) >> 8;

  if (DataTypeSize != 8) {
    assert(false && "Unexpected data type size.");
  }

  ArenaArray<size_t> TensorDimensions(Dimensions);

  for (auto i = 0; i < Dimensions; i++) {
    int Size;
    File.read((char *)&Size, sizeof(int));
    Size = __builtin_bswap32(Size);

    TensorDimensions[i] = static_cast<size_t>(Size);
  }

  Tensor<double> Images(ArenaArray<size_t>(
      {TensorDimensions[1] * TensorDimensions[2], TensorDimensions[0]}));

  for (size_t i = 0; i < Images.Dimensions[1]; i++) {
    for (size_t j = 0; j < Images.Dimensions[0]; j++) {
      unsigned char ByteValue;
      File.read((char *)&ByteValue, sizeof(unsigned char));

      Images[j, i] = (static_cast<double>(ByteValue) / 255.0) + 1e-6;
    }
  }

  return Images;
}

Tensor<double> BuildLabels(Tensor<double> &Images,
                           Tensor<unsigned char> &Labels) {
  Tensor<double> Result(ArenaArray<size_t>({10, Images.Dimensions[1]}));
  for (size_t i = 0; i < Result.Dimensions[1]; i++) {
    auto Label = Labels[i, 0];
    for (size_t j = 0; j < Result.Dimensions[0]; j++) {
      if (j == Label) {
        Result[j, i] = 1.0;
      }
    }
  }

  return Result;
}

void Evaluate(Model<ArenaTensorNode<double>> &Classifier,
              Tensor<double> &TestImages, Tensor<unsigned char> &TestLabels) {
  ArenaWatcher Watcher(GeneralArena);

  auto Labels = BuildLabels(TestImages, TestLabels);
  double CorrectPredictions = 0;

  for (size_t i = 0; i < TestImages.Dimensions[1]; i++) {
    auto Input = ArenaTensorNode<double>(FastGetColumns(TestImages, i, i));
    auto Label = FastGetColumns(Labels, i, i);

    auto Prediction = Classifier.Predict(Input);
    Prediction = Softmax(Prediction);

    if (Max(Prediction.GetValue()) == Max(Label)) {
      CorrectPredictions++;
    }
  }

  std::cout << CorrectPredictions << " correct predictions out of "
            << TestImages.Dimensions[1] << std::endl;
}

void Train(Tensor<double> &TrainingImages,
           Tensor<unsigned char> &TrainingLabels, Tensor<double> &TestImages,
           Tensor<unsigned char> &TestLabels,
           [[maybe_unused]] const std::string &OutputModelName) {
  auto Labels = BuildLabels(TrainingImages, TrainingLabels);

  LinearLayer<ArenaTensorNode<double>> InputLayer(28 * 28, 16);
  ActivationLayer<ArenaTensorNode<double>> ActivationInput(
      SinActivation<double>);
  LinearLayer<ArenaTensorNode<double>> OutputLayer(16, 10);

  auto Classifier = Model<ArenaTensorNode<double>>(
      {&InputLayer, &ActivationInput, &OutputLayer});

  double LearningRate = 1e-4;
  FixedRateScheduler<double> Scheduler(LearningRate);

  AdamWeightDecay<double> Optimizer(Classifier.OptimizationParameters,
                                    &Scheduler, 0.9, 0.999, LearningRate / 100,
                                    1e-6);
  size_t BatchSize = 8;

  ArenaArray<size_t> Indices(TrainingImages.Dimensions[1]);
  for (size_t i = 0; i < Indices.Size; i++) {
    Indices[i] = i;
  }

  for (auto Epoch = 0; Epoch < 10; Epoch++) {
    double AverageLoss = 0;
    double Batches = 0;

    for (size_t i = 0; i < TrainingImages.Dimensions[1]; i += BatchSize) {
      ArenaWatcher Watcher(GeneralArena);

      auto IterationBatchSize = BatchSize;
      auto Index = Indices[i];

      if ((Index + BatchSize) > TrainingImages.Dimensions[1]) {
        IterationBatchSize = (TrainingImages.Dimensions[1] - Index);
      }

      auto Input = ArenaTensorNode<double>(FastGetColumns(
          TrainingImages, Index, (Index + IterationBatchSize) - 1));
      auto Label =
          FastGetColumns(Labels, Index, (Index + IterationBatchSize) - 1);

      auto Prediction = Classifier.Predict(Input);
      auto Loss = CrossEntropy(Prediction, Label);
      Loss = Average(Loss);
      assert(Loss.GetValue().Type == TensorType::Scalar);
      auto Gradient = Tensor<double>(1);
      Loss.Backward(Gradient);

      Optimizer.Optimize();
      Optimizer.ZeroGradients();

      AverageLoss += Loss.GetValue().Data[0];
      Batches++;

#ifndef NDEBUG
      std::cout << "Batch loss: " << Loss.GetValue().Data[0] << " "
                << IterationBatchSize << " " << Index << " " << i << std::endl;
#endif
    }

    std::cout << "Average loss: " << AverageLoss / Batches << std::endl;

    Evaluate(Classifier, TestImages, TestLabels);

    Shuffle(Indices);
  }

  Classifier.WriteToDisk(OutputModelName);
}

int main(int Argc, const char *Argv[]) {
  if (Argc != 6) {
    std::cout << "Usage: " << Argv[0] << " "
              << "<path_to_train_set_images> <path_to_train_set_labels> "
                 "<path_to_test_set_images> <path_to_test_set_labels> "
                 "<output_model_name>"
              << std::endl;
    return 1;
  }

  ArenaWatcher Watcher(GeneralArena);

  auto TrainingImages = ParseMnistImages(Argv[1]);
  auto TrainingLabels = ParseMnistLabels(Argv[2]);

  auto TestImages = ParseMnistImages(Argv[3]);
  auto TestLabels = ParseMnistLabels(Argv[4]);

  Train(TrainingImages, TrainingLabels, TestImages, TestLabels, Argv[5]);

  return 0;
}
