#pragma once
#include <vector>
#include <memory>
#include "DenseLayer.hpp"
#include "Loss.hpp"

class Network {
public:
    void addLayer(int inputSize, int outputSize, ActivationType activation);

    Matrix predict(const Matrix& input);

    // treina por N epochs, imprime a loss a cada 'printEvery' epochs
    void train(const Matrix& inputs, const Matrix& targets,
               int epochs, double learningRate, int printEvery = 100);

private:
    std::vector<std::unique_ptr<DenseLayer>> layers_;
};
