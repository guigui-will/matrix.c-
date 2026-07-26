#include "Loss.hpp"
#include <stdexcept>

double Loss::mse(const Matrix& predicted, const Matrix& target) {
    if (predicted.rows() != target.rows() || predicted.cols() != target.cols())
        throw std::runtime_error("Loss::mse dimensoes incompativeis");
    double sum = 0.0;
    int n = predicted.rows() * predicted.cols();
    for (int i = 0; i < predicted.rows(); ++i)
        for (int j = 0; j < predicted.cols(); ++j) {
            double diff = predicted(i, j) - target(i, j);
            sum += diff * diff;
        }
    return sum / n;
}

Matrix Loss::mseDerivative(const Matrix& predicted, const Matrix& target) {
    int n = predicted.rows() * predicted.cols();
    Matrix result(predicted.rows(), predicted.cols());
    for (int i = 0; i < predicted.rows(); ++i)
        for (int j = 0; j < predicted.cols(); ++j)
            result(i, j) = 2.0 * (predicted(i, j) - target(i, j)) / n;
    return result;
}
