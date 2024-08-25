using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA;

public static class Fusion
{
    internal static long MaxVirtualId = 0;

    public static CudnnTensorDescriptor<T> CreateDescriptor<T>(
        this AbstractTensor<T> tensor)
        where T : unmanaged, IFloatingPoint<T> => 
        new(tensor);

    public static CudnnTensorDescriptor<T> CreateVirtualDescriptor<T>(
        Shape shape)
        where T : unmanaged, IFloatingPoint<T> => 
        new(shape, --MaxVirtualId, isVirtual: true);

    public static CudnnPointwiseOperation<T> Add<T>(
        CudnnTensorDescriptor<T> x,
        CudnnTensorDescriptor<T> b,
        CudnnTensorDescriptor<T> y) 
        where T : unmanaged, IFloatingPoint<T>
    {
        var pw = new CudnnPointwiseOperator<T>(cudnnPointwiseMode_t.CUDNN_POINTWISE_ADD);
        var op = new CudnnPointwiseOperation<T>(pw, x, b, y);
        return op;
    }

    public static CudnnMatMulOperation<T> MatMul<T>(
        CudnnTensorDescriptor<T> a,
        CudnnTensorDescriptor<T> b,
        CudnnTensorDescriptor<T> c) 
        where T : unmanaged, IFloatingPoint<T>
    {
        var mm = new CudnnMatMulOperator<T>();
        var op = new CudnnMatMulOperation<T>(mm, a, b, c);
        return op;
    }
}