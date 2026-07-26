#include "DenseLayer.hpp"
#include <cmath>

DenseLayer::DenseLayer(int inputSize, int outputSize, ActivationType activation)
    : inputSize_(inputSize), outputSize_(outputSize), activation_(activation),
      weights_(Matrix::random(outputSize, inputSize, std::sqrt(2.0 / inputSize))),
      bias_(outputSize, 1, 0.0) {}

Matrix DenseLayer::forward(const Matrix& input) {
    lastInput_ = input;                          // inputSize x batch
    Matrix z = weights_ * input;                 // outputSize x batch
    z = z.addBroadcastCol(bias_);
    lastZ_ = z;
    return Activation::apply(z, activation_);    // outputSize x batch
}

Matrix DenseLayer::backward(const Matrix& dOutput, double learningRate) {
    // dOutput: outputSize x batch  (derivada da loss em relação a saida ativada)
    Matrix dZ = dOutput.hadamard(Activation::derivative(lastZ_, activation_));

    int batchSize = lastInput_.cols();

    Matrix dWeights = (dZ * lastInput_.transpose()) * (1.0 / batchSize);
    Matrix dBias = dZ.sumRows() * (1.0 / batchSize);
    Matrix dInput = weights_.transpose() * dZ;

    weights_ = weights_ - dWeights * learningRate;
    bias_ = bias_ - dBias * learningRate;

    return dInput;
}
