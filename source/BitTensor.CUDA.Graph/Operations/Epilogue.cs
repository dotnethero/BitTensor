using System.Numerics;
using BitTensor.CUDA.Graph.Nodes;

namespace BitTensor.CUDA.Graph;

public interface IEpilogue<T> where T : unmanaged, IFloatingPoint<T>
{
    void Execute(CudaTensor<T> output);
    AbstractNode<T> GetGradient(AbstractNode<T> gradient);
}
