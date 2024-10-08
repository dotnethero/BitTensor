﻿using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Graph.Nodes;

namespace BitTensor.CUDA.Graph;

public static class Ops
{
    // unary
    
    public static CudaNode<T> Reciprocal<T>(
        CudaNode<T> input,
        float scale = 1f)
        where T : unmanaged, IFloatingPoint<T> =>
        new Reciprocal<T>(input, scale);

    public static CudaNode<T> Exp<T>(
        CudaNode<T> input,
        float scale = 1f)
        where T : unmanaged, IFloatingPoint<T> =>
        new Exp<T>(input, scale);
    
    public static CudaNode<T> Log<T>(
        CudaNode<T> input,
        float scale = 1f)
        where T : unmanaged, IFloatingPoint<T> =>
        new Log<T>(input, scale);
    
    public static CudaNode<float> ReLU(
        CudaNode<float> input,
        float alpha = 1f) => 
        new ReLU(input, alpha);
    
    public static CudaNode<float> Softmax(
        CudaNode<float> input,
        CudaBackend backend = CudaBackend.cuTENSOR) => 
        Softmax(input, [^1], backend);

    public static CudaNode<float> Softmax(
        CudaNode<float> input,
        HashSet<Index> axis,
        CudaBackend backend = CudaBackend.cuTENSOR) =>
        backend switch
        {
            CudaBackend.cuTENSOR => new Softmax(input, axis),
            CudaBackend.cuDNN => new SoftmaxCudnn<float>(input, axis),
            _ => throw new ArgumentOutOfRangeException(nameof(backend))
        };

    // reductions
    
    public static CudaNode<T> Sum<T>(
        CudaNode<T> source,
        float scale = 1f,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        new Sum<T>(source, axis: source.Shape.GetAxis(), scale, keepDims);

    public static CudaNode<T> Sum<T>(
        CudaNode<T> source,
        HashSet<Index> axis,
        float scale = 1f,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        new Sum<T>(source, axis, scale, keepDims);
    
    public static CudaNode<T> Max<T>(
        CudaNode<T> source,
        float scale = 1f,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        new Max<T>(source, axis: source.Shape.GetAxis(), scale, keepDims);

    public static CudaNode<T> Max<T>(
        CudaNode<T> source,
        HashSet<Index> axis,
        float scale = 1f,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        new Max<T>(source, axis, scale, keepDims);
    
    public static CudaNode<T> Min<T>(
        CudaNode<T> source,
        float scale = 1f,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        new Min<T>(source, axis: source.Shape.GetAxis(), scale, keepDims);

    public static CudaNode<T> Min<T>(
        CudaNode<T> source,
        HashSet<Index> axis,
        float scale = 1f,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        new Min<T>(source, axis, scale, keepDims);

    // matmul

    public static CudaNode<T> DotProduct<T>(
        CudaNode<T> a,
        CudaNode<T> b,
        float scale = 1f)
        where T : unmanaged, IFloatingPoint<T> =>
        new DotProduct<T>(a, b, scale);

    public static CudaNode<T> MatMul<T>(
        CudaNode<T> a,
        CudaNode<T> b)
        where T : unmanaged, IFloatingPoint<T>
    {
        if (a.IsScalar ||
            b.IsScalar)
            return new Multiply<T>(a, b);

        if (a.IsVector &&
            b.IsVector)
            return new DotProduct<T>(a, b);

        return new MatMul<T>(a, b);
    }

    public static CudaNode<T> Gemm<T>(
        CudaNode<T> a,
        CudaNode<T> b,
        CudaNode<T> c,
        CudaBackend backend = CudaBackend.cuTENSOR)
        where T : unmanaged, IFloatingPoint<T> =>
        backend switch
        {
            CudaBackend.cuTENSOR => new Gemm<T>(a, b, c),
            CudaBackend.cuDNN => new GemmCudnn<T>(a, b, c),
            _ => throw new ArgumentOutOfRangeException(nameof(backend))
        };

    public static CudaNode<float> GemmRelu(
        CudaNode<float> a,
        CudaNode<float> b,
        CudaNode<float> c,
        float alpha) =>
        new GemmReluCudnn<float>(a, b, c, alpha);

}