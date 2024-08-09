// ReSharper disable ConvertToPrimaryConstructor

using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public class LinearLayer : ILayer, IDisposable
{
    public CuTensorNode Weights { get; set; }
    public CuTensorNode Bias { get; set; }
    public Func<CuTensorNode, CuTensorNode> Activation { get; }

    public CuTensorNode[] Parameters => [Weights, Bias];

    public LinearLayer(int inputs, int outputs, Func<CuTensorNode, CuTensorNode> activation)
    {
        Activation = activation;
        Bias = CuTensor.Random.Uniform([outputs]).ToNode();
        Weights = CuTensor.Random.Uniform([inputs, outputs]).ToNode();
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
