#pragma once
#include <vector>
#include <iostream>

class Matrix {
public:
    Matrix(int rows = 0, int cols = 0, double init = 0.0);

    static Matrix random(int rows, int cols, double scale = 1.0);
    static Matrix fromVector(const std::vector<double>& v, bool columnVector = true);

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    double& operator()(int r, int c);
    double operator()(int r, int c) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;      // multiplicação matricial
    Matrix hadamard(const Matrix& other) const;        // multiplicação elemento a elemento
    Matrix operator*(double scalar) const;
    Matrix transpose() const;

    // soma cada coluna de 'other' (bias) a todas as colunas de 'this'
    Matrix addBroadcastCol(const Matrix& colVec) const;

    // soma todas as colunas, retornando um vetor coluna (usado no backprop do bias)
    Matrix sumRows() const;

    void fill(double value);
    void print(const std::string& label = "") const;

    std::vector<double> toVector() const;

private:
    int rows_, cols_;
    std::vector<double> data_;
};
