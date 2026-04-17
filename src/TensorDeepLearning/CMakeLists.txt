add_library(TensorDeepLearningLibrary STATIC 
    Layers/Layer.h Layers/LinearLayer.h 
    Optimizers/Optimizer.h Optimizers/StochasticGradientDescent.h Optimizers/AdaptiveGradient.h Optimizers/Adam.h
    Schedulers/Scheduler.h Schedulers/FixedRateScheduler.h
    Functions/LossFunctions.h Functions/ActivationFunctions.h
    TensorDeepLearning.cpp)

target_include_directories(TensorDeepLearningLibrary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(TensorDeepLearningLibrary TensorAutomaticDifferentiationLibrary)

add_subdirectory(Tests)