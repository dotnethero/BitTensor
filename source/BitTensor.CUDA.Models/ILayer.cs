using System.Numerics;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public interface ILayer<T> where T : unmanaged, IFloatingPoint<T>
{
    CudaWeights<T>[] Parameters { get; }
    CudaNode<T> Compute(CudaNode<T> input);
}

