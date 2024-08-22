using System.Numerics;

namespace BitTensor.CUDA.Graph;

public interface ILayer<T> where T : unmanaged, IFloatingPoint<T>
{
    CudaContext Context { get; }
    CudaWeights<T>[] Parameters { get; }
    CudaNode<T> Compose(CudaNode<T> input);
}
