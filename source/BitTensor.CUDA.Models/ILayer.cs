using System.Numerics;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public interface ILayer<T> where T : unmanaged, IFloatingPoint<T>
{
    CuTensorWeights<T>[] Parameters { get; }
    CuTensorNode<T> Compute(CuTensorNode<T> input);
}

