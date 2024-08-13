// ReSharper disable ConvertToPrimaryConstructor

using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public class LinearLayer : ILayer<float>
{
    public delegate CudaNode<float> ActivationFunction(CudaNode<float> node);

    public CudaWeights<float> Weights { get; set; }
    public CudaWeights<float> Bias { get; set; }
    public ActivationFunction Activation { get; }

    public CudaWeights<float>[] Parameters => [Weights, Bias];

    public LinearLayer(CuContext context, int inputs, int outputs, ActivationFunction activation)
    {
        var weights = context.cuRAND.Normal([inputs, outputs]);
        var bias = context.cuRAND.Normal([outputs]);

        Weights = new CudaWeights<float>(context, weights);
        Bias = new CudaWeights<float>(context, bias);
        Activation = activation;
    }

    public CudaNode<float> Compute(CudaNode<float> input)
    {
        var z = input * Weights + Bias;
        var y = Activation(z);
        return y;
    }
}
