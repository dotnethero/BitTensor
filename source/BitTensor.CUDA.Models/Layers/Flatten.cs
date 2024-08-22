using System.Numerics;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models.Layers;

public class Flatten<T> : ILayer<T> where T : unmanaged, IFloatingPoint<T>
{
    public CudaContext Context { get; }
    public CudaWeights<T>[] Parameters { get; } = [];

    public Flatten(CudaContext context)
    {
        Context = context;
    }

    public CudaNode<T> Compose(CudaNode<T> input)
    {
        var batch = input.Shape.Extents[0];
        var size = input.Shape.Strides[0];
        return input.Reshape([batch, size]);
    }
}