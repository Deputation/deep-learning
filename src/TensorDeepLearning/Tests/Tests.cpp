#include "Functions/LossFunctions.h"
#include "Layers/LinearLayer.h"
#include "Optimizers/AdaptiveGradient.h"
#include "Optimizers/StochasticGradientDescent.h"
#include "Schedulers/FixedRateScheduler.h"

#include <ArenaArray.h>
#include <ArenaTensorNode.h>
#include <ArenaWatcher.h>
#include <GeneratorWrapper.h>
#include <Tensor.h>
#include <TensorKernels.h>
#include <TensorNodes.h>
#include <TensorOperations.h>
#include <TensorUtilities.h>

void TestLinearLayer(size_t MatrixSize, size_t SampleAmount, size_t Epochs,
                     size_t BatchSize) {
  std::cout << "Preparing data" << std::endl;
  std::cout << "Matrix size: " << MatrixSize << std::endl;
  std::cout << "Samples: " << SampleAmount << std::endl;
  std::cout << "Epochs: " << Epochs << std::endl;
  std::cout << "Batch size: " << BatchSize << std::endl;

  assert(SampleAmount > BatchSize);

  ArenaWatcher Watcher(GeneralArena);

  // implicitly add a bias column
  auto WeightMatrix =
      ArenaTensorNode<double>(Tensor<double>({MatrixSize, MatrixSize + 1}, 0));
  for (size_t i = 0; i < WeightMatrix.NodePointer->Value.Data.Size; i++) {
    WeightMatrix[i] = GeneratorWrapper<double>::GetSingleton().Sample();
  }

  Tensor<double> SolutionMatrix({MatrixSize, MatrixSize + 1}, 0);
  size_t k = 0;
  for (size_t i = 0; i < SolutionMatrix.Dimensions[0]; i++) {
    for (size_t j = 0; j < SolutionMatrix.Dimensions[1] - 1; j++) {
      SolutionMatrix[i, j] = (k % 10) + 1;
      k++;
    }

    // the bias column of the solution matrix will be a random number
    SolutionMatrix[i, SolutionMatrix.Dimensions[1] - 1] =
        GeneratorWrapper<double>::GetSingleton().Sample();
  }

  Tensor<double> Inputs(ArenaArray<size_t>({MatrixSize + 1, SampleAmount}));

  for (size_t i = 0; i < MatrixSize; i++) {
    for (size_t j = 0; j < SampleAmount; j++) {
      Inputs[i, j] = GeneratorWrapper<double>::GetSingleton().Sample();

      // pre process inputs: setting the input vector's value to 1
      // makes the weight matrix's extra column act as an added bias vector that
      // can be learned
      Inputs[MatrixSize, j] = 1;
    }
  }

  auto Labels = MatrixMultiplication(SolutionMatrix, Inputs);

  auto LearningRate = 0.5;

  std::cout << "Training" << std::endl;

  for (size_t Epoch = 0; Epoch < Epochs; Epoch++) {
    ArenaWatcher Watcher(GeneralArena);

    auto AverageLoss = Tensor<double>(0);
    auto Batches = 0;

    for (size_t i = 0; i < SampleAmount; i += BatchSize) {
      auto IterationBatchSize = BatchSize;

      if ((BatchSize + i) > SampleAmount) {
        IterationBatchSize = SampleAmount - i;
      }

      Tensor<double> Input =
          FastGetColumns(Inputs, i, i + (IterationBatchSize - 1));
      Tensor<double> Label =
          FastGetColumns(Labels, i, i + (IterationBatchSize - 1));

      auto Prediction = MatrixMultiplication(WeightMatrix, Input);

      auto Loss = SquaredError(Prediction, Label);
      Loss = Average(Loss);
      auto Gradient = Tensor<double>(1);
      Loss.Backward(Gradient);

      AverageLoss += Loss.GetValue();
      Batches++;

      auto WeightGradient =
          WeightMatrix.GetTensorValuePointer()->Gradient * LearningRate;
      WeightMatrix.GetValue() -= WeightGradient;
      WeightMatrix.GetTensorValuePointer()->ZeroGradient();
    }

    AverageLoss /= static_cast<double>(Batches);

    std::cout << "Average loss: ";
    Print(AverageLoss);
  }

  std::cout << "Solution Matrix:" << std::endl;
  Print(SolutionMatrix);

  std::cout << "Sample input:" << std::endl;
  auto Input = FastGetColumns(Inputs, 0, 0);
  Print(Input);

  std::cout << "Learned weights:" << std::endl;
  Print(WeightMatrix.GetValue());
}

void TestLinearLayerClass(size_t MatrixSize, size_t SampleAmount, size_t Epochs,
                          size_t BatchSize) {
  std::cout << "Preparing data" << std::endl;
  std::cout << "Matrix size: " << MatrixSize << std::endl;
  std::cout << "Samples: " << SampleAmount << std::endl;
  std::cout << "Epochs: " << Epochs << std::endl;
  std::cout << "Batch size: " << BatchSize << std::endl;

  assert(SampleAmount > BatchSize);

  ArenaWatcher Watcher(GeneralArena);

  LinearLayer<ArenaTensorNode<double>> Layer(MatrixSize, MatrixSize);

  Tensor<double> SolutionMatrix({MatrixSize, MatrixSize}, 0);
  Tensor<double> SolutionBias(
      {MatrixSize, SampleAmount},
      GeneratorWrapper<double>::GetSingleton().Sample());
  size_t k = 0;
  for (size_t i = 0; i < SolutionMatrix.Dimensions[0]; i++) {
    for (size_t j = 0; j < SolutionMatrix.Dimensions[1]; j++) {
      SolutionMatrix[i, j] = (k % 10) + 1;
      k++;
    }
  }

  Tensor<double> Inputs(ArenaArray<size_t>({MatrixSize, SampleAmount}));

  for (size_t i = 0; i < MatrixSize; i++) {
    for (size_t j = 0; j < SampleAmount; j++) {
      Inputs[i, j] = GeneratorWrapper<double>::GetSingleton().Sample();
    }
  }

  auto Labels = MatrixMultiplication(SolutionMatrix, Inputs) + SolutionBias;

  ArenaArray<ArenaTensorNode<double> *> Parameters(
      Layer.ComputeParameterCount());
  Layer.WriteLayerParameters(Parameters);

  FixedRateScheduler<double> LearningRateScheduler(0.5);
  AdaptiveGradient<double> Optimizer(Parameters, &LearningRateScheduler);

  std::cout << "Training" << std::endl;

  for (size_t Epoch = 0; Epoch < Epochs; Epoch++) {
    ArenaWatcher Watcher(GeneralArena);

    auto AverageLoss = Tensor<double>(0);
    auto Batches = 0;

    for (size_t i = 0; i < SampleAmount; i += BatchSize) {
      auto IterationBatchSize = BatchSize;

      if ((BatchSize + i) > SampleAmount) {
        IterationBatchSize = SampleAmount - i;
      }

      auto Input = ArenaTensorNode<double>(
          FastGetColumns(Inputs, i, i + (IterationBatchSize - 1)));
      auto Label = FastGetColumns(Labels, i, i + (IterationBatchSize - 1));

      auto Prediction = Layer.Predict(Input);

      auto Loss = SquaredError(Prediction, Label);
      Loss = Average(Loss);
      auto Gradient = Tensor<double>(1);
      Loss.Backward(Gradient);

      AverageLoss += Loss.GetValue();
      Batches++;

      Optimizer.Optimize();
      Optimizer.ZeroGradients();
    }

    AverageLoss /= static_cast<double>(Batches);

    std::cout << "Average loss: ";
    Print(AverageLoss);
  }

  std::cout << "Solution Matrix:" << std::endl;
  Print(SolutionMatrix);
  std::cout << "Solution bias:" << std::endl;
  auto BiasColumn = FastGetColumns(SolutionBias, 0, 0);
  Print(BiasColumn);

  std::cout << "Sample input:" << std::endl;
  auto Input = FastGetColumns(Inputs, 0, 0);
  Print(Input);

  std::cout << "Learned weights:" << std::endl;
  Print(Layer.WeightMatrix.GetValue());
  std::cout << "Learned bias:" << std::endl;
  Print(Layer.BiasVector.GetValue());
}

int main([[maybe_unused]] int Argc, [[maybe_unused]] const char *Argv[]) {
  TestLinearLayer(3, 10000, 500, 256);
  TestLinearLayerClass(3, 10000, 500, 256);

  return 0;
}