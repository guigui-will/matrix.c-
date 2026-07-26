#include "Network.hpp"
#include <iostream>

void Network::addLayer(int inputSize, int outputSize, ActivationType activation) {
    layers_.push_back(std::make_unique<DenseLayer>(inputSize, outputSize, activation));
}

Matrix Network::predict(const Matrix& input) {
    Matrix out = input;
    for (auto& layer : layers_) {
        out = layer->forward(out);
    }
    return out;
}

void Network::train(const Matrix& inputs, const Matrix& targets,
                     int epochs, double learningRate, int printEvery) {
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        Matrix output = predict(inputs);

        double loss = Loss::mse(output, targets);
        Matrix dLoss = Loss::mseDerivative(output, targets);

        for (int i = (int)layers_.size() - 1; i >= 0; --i) {
            dLoss = layers_[i]->backward(dLoss, learningRate);
        }

        if (epoch % printEvery == 0 || epoch == 1) {
            std::cout << "Epoch " << epoch << "/" << epochs
                      << " - loss: " << loss << "\n";
        }
    }
}
