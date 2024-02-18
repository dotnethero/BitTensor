# BitTensor - Tensor computation library for .NET

BitTensor is a high-performance, easy-to-use tensor library designed for machine learning applications. It provides a comprehensive set of operations, including arithmetic operations, matrix manipulations, and automatic differentiation, making it ideal for building and training neural networks.

## Features

- **High-Performance Tensor Operations**: Utilize unsafe code for critical sections to enhance performance.
- **Automatic Differentiation**: Support for gradients computation for backpropagation.
- **Model Building Framework**: Easily define and stack neural network layers with `Model` and `SequentialModel`.
- **Support for Broadcasting and Aggregation**: Perform operations on tensors of different shapes efficiently.
- **Customizable**: Define complex operations and custom gradients with support for custom forward and backward functions.

## Installation

Currently, BitTensor is available as a source code repository. Clone the repository to get started:

```bash
git clone https://github.com/yourusername/BitTensor.git
```

## Quick Start

Here's a quick example to get you started with BitTensor:

```csharp
using BitTensor.Core;
using BitTensor.Units;

var linearLayer = new LinearLayer(inputs: 10, outputs: 5, activation: Tensor.Sigmoid);
var input = Tensor.Random.Uniform([1, 10]);
var output = linearLayer.Compute(input);

Console.WriteLine($"Output: {output}");
```

## Examples

### Building a Simple Model

```csharp
var model = Model.Sequential(
[
    new LinearLayer(inputs: 784, outputs: 128, activation: Tensor.ReLU),
    new LinearLayer(inputs: 128, outputs: 10, activation: Tensor.Softmax)
]);

var input = Tensor.Random.Uniform([1, 784]);
var output = model.Compute(input);

Console.WriteLine($"Model output: {output}");
```

### Training the Model

Refer to the `Fit` method in the `Model` class for examples on how to train the model using the provided dataset.

## License

BitTensor is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
