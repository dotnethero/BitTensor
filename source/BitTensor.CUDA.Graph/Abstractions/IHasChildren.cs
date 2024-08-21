using System.Numerics;

namespace BitTensor.CUDA.Graph;

public interface IHasChildren<T> where T : unmanaged, IFloatingPoint<T>
{
    CudaNode<T>[] GetChildren();
}