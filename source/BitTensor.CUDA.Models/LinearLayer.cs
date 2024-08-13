// ReSharper disable ConvertToPrimaryConstructor

using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public class LinearLayer : ILayer<float>
{
    public delegate CuNode<float> ActivationFunction(CuNode<float> node);

    public CuWeights<float> Weights { get; set; }
    public CuWeights<float> Bias { get; set; }
    public ActivationFunction Activation { get; }

    public CuWeights<float>[] Parameters => [Weights, Bias];

    public LinearLayer(CuContext context, int inputs, int outputs, ActivationFunction activation)
    {
        var weights = context.cuRAND.Normal([inputs, outputs]);
        var bias = context.cuRAND.Normal([outputs]);

        Weights = new CuWeights<float>(context, weights);
        Bias = new CuWeights<float>(context, bias);
        Activation = activation;
    }

    public CuNode<float> Compute(CuNode<float> input)
    {
        var z = input * Weights + Bias;
        var y = Activation(z);
        return y;
    }
}
