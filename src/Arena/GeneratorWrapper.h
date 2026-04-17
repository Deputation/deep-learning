#ifndef GENERATOR_WRAPPER_H
#define GENERATOR_WRAPPER_H

#include <random>

template <typename T> class GeneratorWrapper {
private:
  std::random_device RandomDevice;
  std::mt19937 RandomGenerator;
  std::normal_distribution<T> Distribution;

public:
  GeneratorWrapper(T Minimum, T Maximum)
      : RandomGenerator(RandomDevice()), Distribution(Minimum, Maximum) {}

  auto Sample() { return Distribution(RandomGenerator); }

  static GeneratorWrapper<T> &GetSingleton() {
    static GeneratorWrapper<T> Singleton(static_cast<T>(-1e-2),
                                         static_cast<T>(1e-2));
    return Singleton;
  }
};

#endif