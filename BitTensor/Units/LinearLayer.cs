using BitTensor.Core;

namespace BitTensor.Units;

public class LinearLayer : ILayer
{
    public Tensor Weights { get; set; }
    public Tensor Bias { get; set; }
    public ActivationFunction Activation { get; }

    public Tensor[] Parameters => [Weights, Bias];

    public LinearLayer(int inputs, int outputs, ActivationFunction activation)
    {
        Activation = activation;
        Bias = Tensor.Random.Uniform([outputs]);
        Weights = Tensor.Random.Uniform([inputs, outputs]);
    }

    public Tensor Compute(Tensor input)
    {
        var z = Tensor.Matmul(input, Weights) + Bias;
        var y = Activation(z);
        return y;
    }
}