# Rede Neural em C++ (do zero)

Implementação de uma rede neural feita **100% do zero em C++**, sem nenhuma
biblioteca externa de álgebra linear ou machine learning. Toda a matemática
(matrizes, forward pass, backpropagation, gradiente descendente) foi
implementada manualmente. Roda inteiramente no terminal, sem interface gráfica.

## O que tem implementado

- **`Matrix`**: classe própria de álgebra linear (soma, subtração, multiplicação
  matricial, produto de Hadamard, transposta, broadcast de bias).
- **`Activation`**: Sigmoid, ReLU, Tanh, Linear (com suas derivadas).
- **`Loss`**: Erro Quadrático Médio (MSE) e sua derivada.
- **`DenseLayer`**: camada totalmente conectada com forward e backward
  (backpropagation manual, atualização de pesos via gradiente descendente).
- **`Network`**: empilha camadas e roda o loop de treino (epochs, print de loss).
- **`main.cpp`**: exemplo treinando o problema clássico **XOR** — bom teste
  porque só é resolvível com pelo menos uma camada escondida (prova que o
  backprop está correto).

## Estrutura de pastas

```
NeuralNet/
├── include/          # headers (.hpp)
│   ├── Matrix.hpp
│   ├── Activation.hpp
│   ├── Loss.hpp
│   ├── DenseLayer.hpp
│   └── Network.hpp
├── src/               # implementação (.cpp)
│   ├── Matrix.cpp
│   ├── Activation.cpp
│   ├── Loss.cpp
│   ├── DenseLayer.cpp
│   ├── Network.cpp
│   └── main.cpp
├── Makefile
└── README.md
```

## Dependências (bibliotecas pra baixar)

**Nenhuma!** Esse é o ponto: o projeto usa só a **STL** (biblioteca padrão do
C++) — `<vector>`, `<random>`, `<cmath>`, `<iostream>`, `<memory>`. Não precisa
instalar Eigen, nem nenhuma lib de ML.

O que você precisa ter instalado na máquina é só o **compilador e o make**:

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install build-essential
```

### macOS
```bash
xcode-select --install
```
(isso instala o clang, que funciona igual ao g++ pros comandos abaixo)

### Windows
Instalar o [MSYS2](https://www.msys2.org/) e depois, no terminal MSYS2:
```bash
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make
```
Ou usar o **WSL** (Windows Subsystem for Linux) e seguir os passos do Ubuntu acima.

Requisito mínimo: **g++ com suporte a C++17** (qualquer versão recente do GCC
ou Clang, de 2018 pra cá, serve).

## Como compilar e rodar

```bash
cd NeuralNet
make          # compila
./neuralnet   # roda o exemplo do XOR
```

Ou tudo em um comando só:
```bash
make run
```

Pra limpar os binários gerados:
```bash
make clean
```

## Saída esperada

A rede treina por 5000 epochs e deve convergir para algo assim:

```
Entrada (0, 0) -> previsto: ~0.02 | esperado: 0
Entrada (0, 1) -> previsto: ~0.96 | esperado: 1
Entrada (1, 0) -> previsto: ~0.96 | esperado: 1
Entrada (1, 1) -> previsto: ~0.04 | esperado: 0
```

## Próximos passos (se quiser evoluir o projeto)

- [ ] Trocar SGD puro por **Adam** (otimizador mais robusto)
- [ ] Adicionar **mini-batch** (hoje o treino usa todo o dataset de uma vez)
- [ ] Suporte a **Softmax + Cross-Entropy** (pra classificação multi-classe)
- [ ] Carregar o dataset **MNIST** (dígitos manuscritos) e treinar nele
- [ ] Salvar/carregar pesos treinados em arquivo (serialização)
- [ ] Dropout e Batch Normalization

## Por que XOR?

XOR não é linearmente separável — uma rede sem camada escondida (um único
perceptron) não consegue aprendê-lo. Por isso é o teste clássico pra provar
que o backpropagation e as camadas escondidas estão implementados
corretamente.
