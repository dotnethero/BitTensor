using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph;

using Ops = cutensorOperator_t;

public static partial class CuTensorNode
{
    public static CuTensorNode<T> Exp<T>(CuTensorNode<T> a) where T : unmanaged, IFloatingPoint<T>
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

    public static CuTensorNode<T> ReLU<T>(CuTensorNode<T> a) where T : unmanaged, IFloatingPoint<T>
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
    
    public static unsafe CuTensorNode<float> LeakyReLU(CuTensorNode<float> a, float alpha)
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

    public static CuTensorNode<T> Reciprocal<T>(CuTensorNode<T> a) where T : unmanaged, IFloatingPoint<T>
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

    public static CuTensorNode<T> Add<T>(CuTensorNode<T> a, CuTensorNode<T> b, float beta = 1f) where T : unmanaged, IFloatingPoint<T>
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

    public static CuTensorNode<T> ElementwiseProduct<T>(CuTensorNode<T> a, CuTensorNode<T> b, float scale = 1f) where T : unmanaged, IFloatingPoint<T>
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

    public static CuTensorNode<T> DotProduct<T>(CuTensorNode<T> a, CuTensorNode<T> b, float scale = 1f) where T : unmanaged, IFloatingPoint<T>
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

    public static CuTensorNode<T> MatrixProduct<T>(CuTensorNode<T> a, CuTensorNode<T> b) where T : unmanaged, IFloatingPoint<T>
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

    public static CuTensorNode<T> Sum<T>(
        CuTensorNode<T> a,
        float scale = 1f,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        Sum(a, a.Shape.GetOrdinaryAxis().ToHashSet(), scale, keepDims);

    public static CuTensorNode<T> Sum<T>(
        CuTensorNode<T> a,
        HashSet<Index> axis,
        float scale = 1f,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        Reduce(a, axis, Ops.CUTENSOR_OP_ADD, scale, keepDims);

    public static CuTensorNode<T> Max<T>(
        CuTensorNode<T> a,
        HashSet<Index> axis,
        float scale = 1f,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        Reduce(a, axis, Ops.CUTENSOR_OP_MAX, scale, keepDims);

    public static CuTensorNode<T> Min<T>(
        CuTensorNode<T> a,
        HashSet<Index> axis,
        float scale = 1f,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        Reduce(a, axis, Ops.CUTENSOR_OP_MIN, scale, keepDims);

    internal static CuTensorNode<T> Reduce<T>(CuTensorNode<T> a, HashSet<Index> axis, Ops operation, float scale = 1f, bool keepDims = false) where T : unmanaged, IFloatingPoint<T>
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

    public static CuTensorNode<T> Broadcast<T>(CuTensorNode<T> a, Shape shape, float scale = 1f) where T : unmanaged, IFloatingPoint<T>
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
    
    public static CuTensorNode<T> Transpose<T>(CuTensorNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var axis = a.Shape.GetTransposeAxis();
        return Transpose(a, axis);
    }
    
    public static CuTensorNode<T> Transpose<T>(CuTensorNode<T> a, Index[] axis) where T : unmanaged, IFloatingPoint<T>
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

    private static CuTensorNode<T> PadLeft<T>(CuTensorNode<T> node) 
        where T : unmanaged, IFloatingPoint<T> =>
        node.IsVector
            ? node.Reshape([1, ..node.Shape])
            : node;
    
    private static CuTensorNode<T> PadRight<T>(CuTensorNode<T> node)
        where T : unmanaged, IFloatingPoint<T> =>
        node.IsVector
            ? node.Reshape([..node.Shape, 1])
            : node;
    
    private static CuContext GetContext<T>(
        CuTensorNode<T> operand) 
        where T : unmanaged, IFloatingPoint<T> =>
        operand.Context;

    private static CuContext GetContext<T>(
        params CuTensorNode<T>[] operands)
        where T : unmanaged, IFloatingPoint<T> =>
        operands
            .Select(c => c.Context)
            .Distinct()
            .Single();
}

public partial class CuTensorNode<T> where T : unmanaged, IFloatingPoint<T>
{
    public static CuTensorNode<T> operator +(CuTensorNode<T> a, CuTensorNode<T> b) => CuTensorNode.Add(a, b, beta: +1);

    public static CuTensorNode<T> operator -(CuTensorNode<T> a, CuTensorNode<T> b) => CuTensorNode.Add(a, b, beta: -1);

    public static CuTensorNode<T> operator *(CuTensorNode<T> a, CuTensorNode<T> b)
    {
        if (a.IsScalar ||
            b.IsScalar)
            return CuTensorNode.ElementwiseProduct(a, b);

        if (a.IsVector && 
            b.IsVector)
            return CuTensorNode.DotProduct(a, b);

        return CuTensorNode.MatrixProduct(a, b);
    }

    public CuTensorNode<T> Reshape(Shape shape)
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