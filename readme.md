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
1. Create new context
2. Allocate the tensor and create graph nodes
3. Perform the calculation

```csharp
using var context = CudaContext.CreateDefault();

var a = context.cuRAND.Uniform([3, 4]).AsNode(context);
var b = context.cuRAND.Uniform([4, 5]).AsNode(context);
var c = Ops.MatMul(a, b);

CuDebug.WriteExpressionTree(c);

// c:
// t0 = MatMul`1(t1, t2)
// t2 = CudaVariable`1
// t1 = CudaVariable`1

CuDebug.WriteLine(c);

// c(3,5) =
// [[  1.272  1.617  0.944  1.314  1.615 ]
//  [  1.434  1.382  0.651  1.362  1.589 ]
//  [  1.266  1.098  0.868  1.021  1.359 ]]
```

## Examples

### Training a MNIST Model

```csharp
var trainImages = MNIST.ReadImages(@"train-images.idx3-ubyte");
var trainLabels = MNIST.ReadLabels(@"train-labels.idx1-ubyte");
        
const int batchSize = 2048;
const int inputCount = 28 * 28;
const int hiddenCount = 512;
const int outputCount = 10;

using var context = CudaContext.CreateDefault();

var model = Model.Create(
[
    new Flatten<float>(context),
    new Linear(context, inputCount, hiddenCount, Activation.ReLU(0.1f)),
    new Linear(context, hiddenCount, outputCount, Activation.Softmax)
]);

var trainer = Model.Compile(model, Loss.CrossEntropy, trainImages, trainLabels, batchSize);
trainer.Fit(lr: 5e-3f, epochs: 50, trace: true);
```

## License

BitTensor is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
