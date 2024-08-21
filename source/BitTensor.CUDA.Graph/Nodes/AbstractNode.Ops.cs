using System.Numerics;
using BitTensor.CUDA.Graph.Epilogues;

namespace BitTensor.CUDA.Graph.Nodes;

public static class Ops
{
    // unary
    
    public static AbstractNode<T> Exp<T>(
        AbstractNode<T> input,
        float scale)
        where T : unmanaged, IFloatingPoint<T> =>
        new Exp<T>(input, scale);
    
    public static AbstractNode<T> Log<T>(
        AbstractNode<T> input,
        float scale)
        where T : unmanaged, IFloatingPoint<T> =>
        new Log<T>(input, scale);
    
    public static AbstractNode<float> ReLU(
        AbstractNode<float> input,
        float scale) => 
        new ReLU(input, scale);
    
    public static AbstractNode<float> Softmax(
        AbstractNode<float> input) => 
        new Softmax(input, [^1]);

    public static AbstractNode<float> Softmax(
        AbstractNode<float> input,
        HashSet<Index> axis) => 
        new Softmax(input, axis);

    // reductions

    public static AbstractNode<T> Sum<T>(
        AbstractNode<T> source,
        HashSet<Index> axis,
        float scale = 1,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        new Sum<T>(source, axis, scale, keepDims);
    
    public static AbstractNode<T> Max<T>(
        AbstractNode<T> source,
        HashSet<Index> axis,
        float scale = 1,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        new Max<T>(source, axis, scale, keepDims);
    
    public static AbstractNode<T> Min<T>(
        AbstractNode<T> source,
        HashSet<Index> axis,
        float scale = 1,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        new Min<T>(source, axis, scale, keepDims);

    // matmul

    public static AbstractNode<T> MatMul<T>(
        AbstractNode<T> a,
        AbstractNode<T> b)
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

    public static AbstractNode<T> Gemm<T>(
        AbstractNode<T> a,
        AbstractNode<T> b,
        AbstractNode<T> c,
        IEpilogue<T> epilogue)
        where T : unmanaged, IFloatingPoint<T> =>
        new Gemm<T>(a, b, c, epilogue);
}
