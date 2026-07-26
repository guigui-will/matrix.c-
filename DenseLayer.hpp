#pragma once
#include "Matrix.hpp"
#include "Activation.hpp"

// Camada totalmente conectada: entrada -> saida, com pesos, bias e ativação.
// As matrizes de entrada/saida usam colunas como amostras (cada coluna = 1 exemplo do batch).
class DenseLayer {
public:
    DenseLayer(int inputSize, int outputSize, ActivationType activation);

    Matrix forward(const Matrix& input);
    // recebe dLoss/dOutput e devolve dLoss/dInput, atualizando pesos e bias
    Matrix backward(const Matrix& dOutput, double learningRate);

private:
    int inputSize_, outputSize_;
    ActivationType activation_;

    Matrix weights_;   // outputSize x inputSize
    Matrix bias_;      // outputSize x 1

    // cache para o backward
    Matrix lastInput_;
    Matrix lastZ_;     // pre-ativação
};
