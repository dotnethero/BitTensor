using System.Numerics;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public interface ILayer<T> where T : unmanaged, IFloatingPoint<T>
{
    CuWeights<T>[] Parameters { get; }
    CuNode<T> Compute(CuNode<T> input);
}

