using System.Numerics;

namespace BitTensor.CUDA.Graph;

public interface IEpilogue<T> where T : unmanaged, IFloatingPoint<T>
{
    void ExecuteInplace(CudaTensor<T> output);
    CudaNode<T> GetGradient(CudaNode<T> gradient);
}
