using System.Numerics;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public interface ILayer<T> where T : unmanaged, INumberBase<T>
{
    CuTensorWeights<T>[] Parameters { get; }
    CuTensorNode<T> Compute(CuTensorNode<T> input);
}

