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

void Evaluate(Model<Tensor<double>> &Classifier, Tensor<double> &TestImages,
              Tensor<unsigned char> &TestLabels) {
  ArenaWatcher Watcher(GeneralArena);

  auto Labels = BuildLabels(TestImages, TestLabels);
  double CorrectPredictions = 0;

  for (size_t i = 0; i < TestImages.Dimensions[1]; i++) {
    auto Input = FastGetColumns(TestImages, i, i);
    auto Label = FastGetColumns(Labels, i, i);

    auto Prediction = Classifier.Predict(Input);
    Prediction = Softmax(Prediction);

    if (Max(Prediction) == Max(Label)) {
      CorrectPredictions++;
    }
  }

  std::cout << CorrectPredictions << " correct predictions out of "
            << TestImages.Dimensions[1] << std::endl;
}

int main(int Argc, const char *Argv[]) {
  if (Argc != 4) {
    std::cout
        << "Usage: " << Argv[0] << " "
        << "<model_path> <path_to_test_set_images> <path_to_test_set_labels>"
        << std::endl;
    return 1;
  }

  ArenaWatcher Watcher(GeneralArena);

  LinearLayer<Tensor<double>> InputLayer(28 * 28, 16);
  ActivationLayer<Tensor<double>> ActivationInput(
      SinNoGradient<double>);
  LinearLayer<Tensor<double>> OutputLayer(16, 10);
  auto Classifier = Model<Tensor<double>>({&InputLayer, &ActivationInput, &OutputLayer});
  Classifier.LoadFromDisk(Argv[1]);

  auto TestImages = ParseMnistImages(Argv[2]);
  auto TestLabels = ParseMnistLabels(Argv[3]);

  Evaluate(Classifier, TestImages, TestLabels);

  return 0;
}