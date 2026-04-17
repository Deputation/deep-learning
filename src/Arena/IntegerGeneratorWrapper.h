#ifndef INTEGER_GENERATOR_WRAPPER_H
#define INTEGER_GENERATOR_WRAPPER_H

#include <limits>
#include <random>

template <typename T> class IntegerGeneratorWrapper {
private:
  std::random_device RandomDevice;
  std::mt19937 RandomGenerator;
  std::uniform_int_distribution<T> Distribution;

public:
  IntegerGeneratorWrapper(T Minimum, T Maximum)
      : RandomGenerator(RandomDevice()), Distribution(Minimum, Maximum) {}

  auto Sample() { return Distribution(RandomGenerator); }

  static IntegerGeneratorWrapper<T> &GetSingleton() {
    static IntegerGeneratorWrapper<T> Singleton(std::numeric_limits<T>::min(),
                                                std::numeric_limits<T>::max());
    return Singleton;
  }
};

#endif