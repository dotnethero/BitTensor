using System.Numerics;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public sealed class Sequential<T> : ILayer<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly ILayer<T>[] Layers;

    public CudaContext Context { get; }
    public CudaWeights<T>[] Parameters => Layers
        .SelectMany(x => x.Parameters)
        .ToArray();

    internal Sequential(ILayer<T>[] layers)
    {
        Context = layers[0].Context;
        Layers = layers;
    }

    public CudaNode<T> Compose(CudaNode<T> input) => 
        Layers.Aggregate(input, (activation, layer) => layer.Compose(activation));
}