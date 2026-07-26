#pragma once
#include "Matrix.hpp"

enum class ActivationType {
    Sigmoid,
    ReLU,
    Tanh,
    Linear
};

class Activation {
public:
    static Matrix apply(const Matrix& z, ActivationType type);
    static Matrix derivative(const Matrix& z, ActivationType type);
};
