using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA;

internal static class Fusion
{
    internal static long MaxVirtualId;

    public static CudnnTensorDescriptor<T> CreateDescriptor<T>(
        this AbstractTensor<T> tensor)
        where T : unmanaged, IFloatingPoint<T> => 
        new(tensor);

    public static CudnnTensorDescriptor<T> CreateVirtualDescriptor<T>(
        Shape shape)
        where T : unmanaged, IFloatingPoint<T> => 
        new(shape, --MaxVirtualId, isVirtual: true);

    // reduction
    
    public static CudnnReductionOperation<T> Sum<T>(
        CudnnTensorDescriptor<T> x,
        CudnnTensorDescriptor<T> y) 
        where T : unmanaged, IFloatingPoint<T>
    {
        var descriptor = new CudnnReductionOperator<T>(cudnnReduceTensorOp_t.CUDNN_REDUCE_TENSOR_ADD);
        var operation = new CudnnReductionOperation<T>(descriptor, x, y);
        return operation;
    }

    public static CudnnReductionOperation<T> Min<T>(
        CudnnTensorDescriptor<T> x,
        CudnnTensorDescriptor<T> y) 
        where T : unmanaged, IFloatingPoint<T>
    {
        var descriptor = new CudnnReductionOperator<T>(cudnnReduceTensorOp_t.CUDNN_REDUCE_TENSOR_MIN);
        var operation = new CudnnReductionOperation<T>(descriptor, x, y);
        return operation;
    }
    
    public static CudnnReductionOperation<T> Max<T>(
        CudnnTensorDescriptor<T> x,
        CudnnTensorDescriptor<T> y) 
        where T : unmanaged, IFloatingPoint<T>
    {
        var descriptor = new CudnnReductionOperator<T>(cudnnReduceTensorOp_t.CUDNN_REDUCE_TENSOR_MAX);
        var operation = new CudnnReductionOperation<T>(descriptor, x, y);
        return operation;
    }

    // unary

    public static CudnnPointwiseOperation<T> Exp<T>(
        CudnnTensorDescriptor<T> x,
        CudnnTensorDescriptor<T> y) 
        where T : unmanaged, IFloatingPoint<T>
    {
        var pw = new CudnnPointwiseOperator<T>(cudnnPointwiseMode_t.CUDNN_POINTWISE_EXP);
        var op = new CudnnPointwiseOperation<T>(pw, x, y);
        return op;
    }

    public static CudnnPointwiseOperation<T> ReLU<T>(
        CudnnTensorDescriptor<T> x,
        CudnnTensorDescriptor<T> y,
        float alpha) 
        where T : unmanaged, IFloatingPoint<T>
    {
        var pw = new CudnnPointwiseOperator<T>(cudnnPointwiseMode_t.CUDNN_POINTWISE_RELU_FWD, lowerClipSlope: alpha);
        var op = new CudnnPointwiseOperation<T>(pw, x, y);
        return op;
    }
    
    // binary

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
    
    public static CudnnPointwiseOperation<T> Subtract<T>(
        CudnnTensorDescriptor<T> x,
        CudnnTensorDescriptor<T> b,
        CudnnTensorDescriptor<T> y) 
        where T : unmanaged, IFloatingPoint<T>
    {
        var pw = new CudnnPointwiseOperator<T>(cudnnPointwiseMode_t.CUDNN_POINTWISE_SUB);
        var op = new CudnnPointwiseOperation<T>(pw, x, b, y);
        return op;
    }
    
    public static CudnnPointwiseOperation<T> Multiply<T>(
        CudnnTensorDescriptor<T> x,
        CudnnTensorDescriptor<T> b,
        CudnnTensorDescriptor<T> y) 
        where T : unmanaged, IFloatingPoint<T>
    {
        var pw = new CudnnPointwiseOperator<T>(cudnnPointwiseMode_t.CUDNN_POINTWISE_MUL);
        var op = new CudnnPointwiseOperation<T>(pw, x, b, y);
        return op;
    }
    
    public static CudnnPointwiseOperation<T> Divide<T>(
        CudnnTensorDescriptor<T> x,
        CudnnTensorDescriptor<T> b,
        CudnnTensorDescriptor<T> y) 
        where T : unmanaged, IFloatingPoint<T>
    {
        var pw = new CudnnPointwiseOperator<T>(cudnnPointwiseMode_t.CUDNN_POINTWISE_DIV);
        var op = new CudnnPointwiseOperation<T>(pw, x, b, y);
        return op;
    }

    // matmul

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