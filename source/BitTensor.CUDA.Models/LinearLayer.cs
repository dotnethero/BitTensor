// ReSharper disable ConvertToPrimaryConstructor

using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public class LinearLayer : ILayer
{
    public CuTensorWeights Weights { get; set; }
    public CuTensorWeights Bias { get; set; }
    public Func<CuTensorNode, CuTensorNode> Activation { get; }

    public CuTensorWeights[] Parameters => [Weights, Bias];

    public LinearLayer(CuContext context, int inputs, int outputs, Func<CuTensorNode, CuTensorNode> activation)
    {
        Activation = activation;
        Bias = new CuTensorWeights(context.cuRAND.Normal([outputs]));
        Weights = new CuTensorWeights(context.cuRAND.Normal([inputs, outputs]));
    }

    public CuTensorNode Compute(CuTensorNode input)
    {
        var z = input * Weights + Bias;
        var y = Activation(z);
        return y;
    }
}
