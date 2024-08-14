using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph;

using Ops = cutensorOperator_t;

public static partial class CuNode
{
    public static CudaNode<T> Sum<T>(
        CudaNode<T> a,
        float scale = 1f,
        bool keepDims = false)
        where T : unmanaged, IFloatingPoint<T> =>
        Sum(a, a.Shape.GetOrdinaryAxis().ToHashSet(), scale, keepDims);

    public static CudaNode<T> Sum<T>(
        CudaNode<T> a,
        HashSet<Index> axis,
        float scale = 1f,
        bool keepDims = false)
        where T : unmanaged, IFloatingPoint<T> =>
        Reduce(a, axis, Ops.CUTENSOR_OP_ADD, scale, keepDims);

    public static CudaNode<T> Max<T>(
        CudaNode<T> a,
        HashSet<Index> axis,
        float scale = 1f,
        bool keepDims = false)
        where T : unmanaged, IFloatingPoint<T> =>
        Reduce(a, axis, Ops.CUTENSOR_OP_MAX, scale, keepDims);

    public static CudaNode<T> Min<T>(
        CudaNode<T> a,
        HashSet<Index> axis,
        float scale = 1f,
        bool keepDims = false)
        where T : unmanaged, IFloatingPoint<T> =>
        Reduce(a, axis, Ops.CUTENSOR_OP_MIN, scale, keepDims);

    private static CudaNode<T> Reduce<T>(CudaNode<T> a, HashSet<Index> axis, Ops operation, float scale = 1f, bool keepDims = false) where T : unmanaged, IFloatingPoint<T>
    {
        var context = a.Context;
        var inputShape = a.Shape;
        var outputShape = inputShape.Reduce(axis, keepDims);
        var plan = context.cuTENSOR.CreateReductionPlan<T>(inputShape, outputShape, axis, operation, keepDims);
        return new(
            context,
            outputShape,
            children: [a],
            forward: (output) => plan.Execute(a, output, scale),
            backward: (grad, _) => [Broadcast(grad, inputShape, scale)]);
    }
}