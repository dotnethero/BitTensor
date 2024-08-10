// ReSharper disable ConvertToPrimaryConstructor

using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Models;

public class LinearLayer : ILayer, IDisposable
{
    public CuTensorWeight Weights { get; set; }
    public CuTensorWeight Bias { get; set; }
    public Func<CuTensorNode, CuTensorNode> Activation { get; }

    public CuTensorWeight[] Parameters => [Weights, Bias];

    public LinearLayer(CuTensorContext context, int inputs, int outputs, Func<CuTensorNode, CuTensorNode> activation)
    {
        Activation = activation;
        Bias = new CuTensorWeight(context, CuTensor.Random.Normal([outputs]));
        Weights = new CuTensorWeight(context, CuTensor.Random.Normal([inputs, outputs]));
    }

    public CuTensorNode Compute(CuTensorNode input)
    {
        var z = input * Weights + Bias;
        var y = Activation(z);
        return y;
    }

    public void Dispose()
    {
        Weights.Dispose();
        Bias.Dispose();
    }
}
