// ReSharper disable ConvertToPrimaryConstructor

using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public class LinearLayer : ILayer<float>
{
    public delegate CuTensorNode<float> ActivationFunction(CuTensorNode<float> node);

    public CuTensorWeights<float> Weights { get; set; }
    public CuTensorWeights<float> Bias { get; set; }
    public ActivationFunction Activation { get; }

    public CuTensorWeights<float>[] Parameters => [Weights, Bias];

    public LinearLayer(CuContext context, int inputs, int outputs, ActivationFunction activation)
    {
        var weights = context.cuRAND.Normal([inputs, outputs]);
        var bias = context.cuRAND.Normal([outputs]);

        Weights = new CuTensorWeights<float>(context, weights);
        Bias = new CuTensorWeights<float>(context, bias);
        Activation = activation;
    }

    public CuTensorNode<float> Compute(CuTensorNode<float> input)
    {
        var z = input * Weights + Bias;
        var y = Activation(z);
        return y;
    }
}
