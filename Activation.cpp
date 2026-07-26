#include "Activation.hpp"
#include <cmath>

static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

Matrix Activation::apply(const Matrix& z, ActivationType type) {
    Matrix out(z.rows(), z.cols());
    for (int i = 0; i < z.rows(); ++i) {
        for (int j = 0; j < z.cols(); ++j) {
            double x = z(i, j);
            double val;
            switch (type) {
                case ActivationType::Sigmoid: val = sigmoid(x); break;
                case ActivationType::ReLU:    val = x > 0 ? x : 0.0; break;
                case ActivationType::Tanh:    val = std::tanh(x); break;
                case ActivationType::Linear:  val = x; break;
                default: val = x;
            }
            out(i, j) = val;
        }
    }
    return out;
}

// Recebe z (pre-ativação) e devolve a derivada da ativação em z
Matrix Activation::derivative(const Matrix& z, ActivationType type) {
    Matrix out(z.rows(), z.cols());
    for (int i = 0; i < z.rows(); ++i) {
        for (int j = 0; j < z.cols(); ++j) {
            double x = z(i, j);
            double val;
            switch (type) {
                case ActivationType::Sigmoid: {
                    double s = sigmoid(x);
                    val = s * (1.0 - s);
                    break;
                }
                case ActivationType::ReLU:
                    val = x > 0 ? 1.0 : 0.0;
                    break;
                case ActivationType::Tanh: {
                    double t = std::tanh(x);
                    val = 1.0 - t * t;
                    break;
                }
                case ActivationType::Linear:
                    val = 1.0;
                    break;
                default:
                    val = 1.0;
            }
            out(i, j) = val;
        }
    }
    return out;
}
