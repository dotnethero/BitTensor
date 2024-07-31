using BitTensor.Abstractions;
using BitTensor.Core;

namespace BitTensor.Units;

public class LinearLayerInv : ILayer
{
    public Tensor Weights { get; set; }
    public Tensor Bias { get; set; }
    public Activation<Tensor> Activation { get; }

    public Tensor[] Parameters => [Weights, Bias];

    public LinearLayerInv(int inputs, int outputs, Activation<Tensor> activation)
    {
        Activation = activation;
        Bias = Tensor.Random.Uniform([outputs, 1]);
        Weights = Tensor.Random.Uniform([outputs, inputs]);
    }

    public Tensor Compute(Tensor input)
    {
        var z = Tensor.Matmul(Weights, input) + Bias;
        var y = Activation(z);
        return y;
    }
}