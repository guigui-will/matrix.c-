#include <iostream>
#include <iomanip>
#include "Network.hpp"

int main() {
    std::cout << "=== Rede Neural em C++ - Treinando XOR ===\n\n";

    // Dataset XOR: 4 amostras, 2 features cada.
    // Convenção: colunas = amostras, linhas = features.
    Matrix inputs(2, 4);
    // amostra 0: (0,0)  amostra 1: (0,1)  amostra 2: (1,0)  amostra 3: (1,1)
    inputs(0,0)=0; inputs(1,0)=0;
    inputs(0,1)=0; inputs(1,1)=1;
    inputs(0,2)=1; inputs(1,2)=0;
    inputs(0,3)=1; inputs(1,3)=1;

    Matrix targets(1, 4);
    targets(0,0)=0;
    targets(0,1)=1;
    targets(0,2)=1;
    targets(0,3)=0;

    Network net;
    net.addLayer(2, 8, ActivationType::Tanh);
    net.addLayer(8, 1, ActivationType::Sigmoid);

    net.train(inputs, targets, /*epochs=*/5000, /*learningRate=*/0.5, /*printEvery=*/500);

    std::cout << "\n=== Resultado final ===\n";
    Matrix predictions = net.predict(inputs);
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < 4; ++i) {
        std::cout << "Entrada (" << inputs(0,i) << ", " << inputs(1,i) << ") -> "
                  << "previsto: " << predictions(0,i)
                  << " | esperado: " << targets(0,i) << "\n";
    }

    return 0;
}
