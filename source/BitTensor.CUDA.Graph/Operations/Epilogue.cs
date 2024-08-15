using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph;

public interface IEpilogue<T> where T : unmanaged, IFloatingPoint<T>
{
    void Execute(CudaTensor<T> output);
    CudaNode<T> Propagate(CudaNode<T> gradient);
}

public sealed class Identity : IEpilogue<float>
{
    public void Execute(CudaTensor<float> output) 
    {
    }

    public CudaNode<float> Propagate(CudaNode<float> gradient) => gradient;
}

public sealed unsafe class ReLU(float alpha = 0) : IEpilogue<float>
{
    public void Execute(CudaTensor<float> output) => 
        Kernels.LeakyReLU(output.Size, output.Pointer, output.Pointer, alpha);

    public CudaNode<float> Propagate(CudaNode<float> gradient) => 
        Ops.LeakyReLU(gradient, alpha);
}
