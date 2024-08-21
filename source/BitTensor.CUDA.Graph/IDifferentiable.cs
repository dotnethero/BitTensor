using System.Numerics;

namespace BitTensor.CUDA.Graph;

public interface IDifferentiable<T> where T : unmanaged, IFloatingPoint<T>
{
    CudaNode<T>[] Propagate(CudaNode<T> gradient);
    CudaNode<T>[] GetChildren();
}
