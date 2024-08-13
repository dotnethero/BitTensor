using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph;

using Ops = cutensorOperator_t;

public static class CuNode
{
    public static CudaNode<T> Exp<T>(CudaNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var context = GetContext(a);
        var output = context.Allocate<T>(a.Shape);
        var plan = context.cuTENSOR.CreateUnaryPlan<T>(a, output, Ops.CUTENSOR_OP_EXP);
        return new(
            context,
            output,
            children: [a],
            forward: () => plan.Execute(a, output, gamma: 0),
            backward: (grad, self) => [ElementwiseProduct(grad, self)]);
    }

    public static CudaNode<T> ReLU<T>(CudaNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var context = GetContext(a);
        var output = context.Allocate<T>(a.Shape);
        var plan = context.cuTENSOR.CreateUnaryPlan<T>(a, output, Ops.CUTENSOR_OP_RELU);
        return new(
            context,
            output,
            children: [a],
            forward: () => plan.Execute(a, output, gamma: 0),
            backward: (grad, _) => [ReLU(grad)]);
    }
    
    public static unsafe CudaNode<float> LeakyReLU(CudaNode<float> a, float alpha)
    {
        var context = GetContext(a);
        var output = context.Allocate<float>(a.Shape);
        return new(
            context,
            output,
            children: [a],
            forward: () => Kernels.LeakyReLU(a.Size, a.Pointer, output.Pointer, alpha),
            backward: (grad, _) => [LeakyReLU(grad, alpha)]);
    }

    public static CudaNode<T> Reciprocal<T>(CudaNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var context = GetContext(a);
        var output = context.Allocate<T>(a.Shape);
        var plan = context.cuTENSOR.CreateUnaryPlan<T>(a, output, Ops.CUTENSOR_OP_RCP);
        return new(
            context,
            output,
            children: [a],
            forward: () => plan.Execute(a, output, gamma: 0),
            backward: (_, self) => [ElementwiseProduct(self, self, -1)]);
    }

    public static CudaNode<T> Add<T>(CudaNode<T> a, CudaNode<T> b, float beta = 1f) where T : unmanaged, IFloatingPoint<T>
    {
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var context = GetContext(a, b);
        var output = context.Allocate<T>(shape);
        var plan = context.cuTENSOR.CreateAddPlan<T>(a, b, output);
        return new(
            context,
            output,
            children: [a, b],
            forward: () => plan.Execute(a, b, output, beta: beta),
            backward: (grad, _) =>
            {
                var adims = Shapes.GetBroadcastedAxis(a.Shape, grad.Shape);
                var bdims = Shapes.GetBroadcastedAxis(b.Shape, grad.Shape);
                return
                [
                    Sum(grad, axis: adims, scale: 1f).Reshape(a.Shape),
                    Sum(grad, axis: bdims, scale: beta).Reshape(b.Shape)
                ];
            });
    }

    public static CudaNode<T> ElementwiseProduct<T>(CudaNode<T> a, CudaNode<T> b, float scale = 1f) where T : unmanaged, IFloatingPoint<T>
    {
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var context = GetContext(a, b);
        var output = context.Allocate<T>(shape);
        var plan = context.cuTENSOR.CreateMultiplyPlan<T>(a, b, output);
        return new(
            context,
            output,
            children: [a, b],
            forward: () => plan.Execute(a, b, output, alpha: scale),
            backward: (grad, _) =>
            {
                var agrad = ElementwiseProduct(grad, b, scale);
                var bgrad = ElementwiseProduct(grad, a, scale);
                var adims = Shapes.GetBroadcastedAxis(a.Shape, agrad.Shape);
                var bdims = Shapes.GetBroadcastedAxis(b.Shape, bgrad.Shape);
                return
                [
                    Sum(agrad, axis: adims).Reshape(a.Shape),
                    Sum(bgrad, axis: bdims).Reshape(b.Shape)
                ];
            });
    }

    public static CudaNode<T> DotProduct<T>(CudaNode<T> a, CudaNode<T> b, float scale = 1f) where T : unmanaged, IFloatingPoint<T>
    {
        Shapes.EnsureAreEqual(a.Shape, b.Shape);
        var context = GetContext(a, b);
        var output = context.Allocate<T>([]);
        var plan = context.cuTENSOR.CreateContractionPlan<T>(a, b, output);
        return new(
            context,
            output,
            children: [a, b],
            forward: () => plan.Execute(a, b, output, alpha: scale),
            backward: (grad, _) => [grad * b, a * grad]); // TODO: scale!
    }

    public static CudaNode<T> MatrixProduct<T>(CudaNode<T> a, CudaNode<T> b) where T : unmanaged, IFloatingPoint<T>
    {
        var shape = Shapes.BroadcastMatrixProduct(a.Shape, b.Shape); // desired shape
        var context = GetContext(a, b);
        var output = context.Allocate<T>(shape); // true output

        var modA = PadLeft(a);
        var modB = PadRight(b);
        var modShape = Shapes.BroadcastMatrixProduct(modA.Shape, modB.Shape); // padded shape
        var modOutput = output.Reshape(modShape); // padded output
        var plan = context.cuTENSOR.CreateMatMulPlan<T>(modA, modB, modOutput);

        return new(
            context,
            output,
            children: [a, b],
            forward: () => plan.Execute(modA, modB, modOutput),
            backward: (grad, _) =>
            {
                var gpad = grad.Reshape(modShape);
                var da = gpad * Transpose(modB);
                var db = Transpose(modA) * gpad;
                
                var adims = Shapes.GetBroadcastedAxis(modA.Shape, da.Shape);
                var bdims = Shapes.GetBroadcastedAxis(modB.Shape, db.Shape);
                var agrad = Sum(da, axis: adims).Reshape(a.Shape);
                var bgrad = Sum(db, axis: bdims).Reshape(b.Shape);
                return [agrad, bgrad];
            });
    }

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

    internal static CudaNode<T> Reduce<T>(CudaNode<T> a, HashSet<Index> axis, Ops operation, float scale = 1f, bool keepDims = false) where T : unmanaged, IFloatingPoint<T>
    {
        var context = GetContext(a);
        var shape = a.Shape.Reduce(axis, keepDims);
        var output = context.Allocate<T>(shape);
        var plan = context.cuTENSOR.CreateReductionPlan<T>(a, output, axis, operation, keepDims);
        return new(
            context,
            output,
            children: [a],
            forward: () => plan.Execute(a, output, scale),
            backward: (grad, _) => [Broadcast(grad, a.Shape, scale)]);
    }

    public static CudaNode<T> Broadcast<T>(CudaNode<T> a, Shape shape, float scale = 1f) where T : unmanaged, IFloatingPoint<T>
    {
        if (!a.Shape.CanBroadcastTo(shape))
            throw new InvalidOperationException($"Can't broadcast {a.Shape} to {shape}");

        var context = GetContext(a);
        var output = context.Allocate<T>(shape);
        var axis = Shapes.GetBroadcastedAxis(a.Shape, shape);
        var plan = context.cuTENSOR.CreateBroadcastPlan<T>(a, output);
        return new(
            context,
            output,
            children: [a],
            forward: () => plan.Execute(a, output, alpha: scale, gamma: 0),
            backward: (grad, _) => [Sum(grad, axis, scale: scale)]); // TODO: Verify!
    }
    
    public static CudaNode<T> Transpose<T>(CudaNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var axis = a.Shape.GetTransposeAxis();
        return Transpose(a, axis);
    }
    
    public static CudaNode<T> Transpose<T>(CudaNode<T> a, Index[] axis) where T : unmanaged, IFloatingPoint<T>
    {
        if (axis.Length != a.Dimensions)
            throw new InvalidOperationException($"Axis {axis.ToText()} is not valid argument for {a.Shape} shape tensor");

        if (!a.Shape.AxisAreUnique(axis))
            throw new InvalidOperationException($"Axis {axis.ToText()} does not contain all axes for {a.Shape} shape tensor");

        var shape = a.Shape.Transpose(axis);
        var context = GetContext(a);
        var output = context.Allocate<T>(shape);
        var plan = context.cuTENSOR.CreatePermutationPlan<T>(a, output, axis);
        return new(
            context,
            output,
            children: [a],
            forward: () => plan.Execute(a, output),
            backward: (grad, _) => [Transpose(grad, axis)]); // TODO: Verify!
    }
    
    public static CudaNode<T> Softmax<T>(CudaNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var max = Max(a, [^1], keepDims: true);
        var ex = Exp(a - max);
        var sumex = Sum(ex, [^1], keepDims: true);
        return ElementwiseProduct(ex, Reciprocal(sumex));
    }

    private static CudaNode<T> PadLeft<T>(CudaNode<T> node) 
        where T : unmanaged, IFloatingPoint<T> =>
        node.IsVector
            ? node.Reshape([1, ..node.Shape])
            : node;
    
    private static CudaNode<T> PadRight<T>(CudaNode<T> node)
        where T : unmanaged, IFloatingPoint<T> =>
        node.IsVector
            ? node.Reshape([..node.Shape, 1])
            : node;
    
    private static CuContext GetContext<T>(
        CudaNode<T> operand) 
        where T : unmanaged, IFloatingPoint<T> =>
        operand.Context;

    private static CuContext GetContext<T>(
        params CudaNode<T>[] operands)
        where T : unmanaged, IFloatingPoint<T> =>
        operands
            .Select(c => c.Context)
            .Distinct()
            .Single();
}

public partial class CudaNode<T> where T : unmanaged, IFloatingPoint<T>
{
    public static CudaNode<T> operator +(CudaNode<T> a, CudaNode<T> b) => Graph.CuNode.Add(a, b, beta: +1);

    public static CudaNode<T> operator -(CudaNode<T> a, CudaNode<T> b) => Graph.CuNode.Add(a, b, beta: -1);

    public static CudaNode<T> operator *(CudaNode<T> a, CudaNode<T> b)
    {
        if (a.IsScalar ||
            b.IsScalar)
            return Graph.CuNode.ElementwiseProduct(a, b);

        if (a.IsVector && 
            b.IsVector)
            return Graph.CuNode.DotProduct(a, b);

        return Graph.CuNode.MatrixProduct(a, b);
    }

    public CudaNode<T> Reshape(Shape shape)
    {
        if (shape.ArraySize != Size)
            throw new InvalidOperationException($"Can't reshape {Shape} into {shape}");

        var output = Tensor.Reshape(shape); // no allocation
        return new(
            Context,
            output,
            children: [this],
            forward: () => {},
            backward: (grad, _) => [grad.Reshape(Shape)]);
    }
}