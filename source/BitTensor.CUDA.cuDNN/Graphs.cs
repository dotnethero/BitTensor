using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA;

public static class Graphs
{
    public static ICudnnGraph Gemm<T>(
        this CudnnContext context,
        AbstractTensor<T> a,
        AbstractTensor<T> b,
        AbstractTensor<T> c,
        AbstractTensor<T> x) 
        where T : unmanaged, IFloatingPoint<T> =>
        new CudnnGemmGraph<T>(context, a, b, c, x);
}
