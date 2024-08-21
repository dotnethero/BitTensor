using System.Numerics;
using BitTensor.CUDA.Graph.Nodes;

namespace BitTensor.CUDA.Graph.Epilogues;

public interface IEpilogue<T> where T : unmanaged, IFloatingPoint<T>
{
    void ExecuteInplace(CudaTensor<T> output);
    AbstractNode<T> GetGradient(AbstractNode<T> gradient);
}
