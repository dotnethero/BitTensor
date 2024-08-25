using System.Numerics;

namespace BitTensor.CUDA.Wrappers;

public interface ICudnnPlan : IDisposable
{
    void Execute<T>(CudnnVariantPack<T> pack) where T : unmanaged, IFloatingPoint<T>;
    long GetWorkspaceSize();
}
