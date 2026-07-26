#pragma once
#include "Matrix.hpp"

class Loss {
public:
    // Erro quadrático médio
    static double mse(const Matrix& predicted, const Matrix& target);
    static Matrix mseDerivative(const Matrix& predicted, const Matrix& target);
};
