using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph;

public static class CuTensorNode
{
    public static CuTensorNode<T> Add<T>(CuTensorNode<T> a, CuTensorNode<T> b, float beta = 1f) where T : unmanaged, INumberBase<T>
    {
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var context = GetContext(a, b);
        var output = context.Allocate<T>(shape);
        var plan = context.CreateAddPlan<T>(a, b, output);
        return new(
            output,
            children: [a, b],
            forward: () => plan.Execute(a, b, output, beta),
            backward: (grad, _) =>
            {
                var adims = Shapes.GetBroadcastedAxis(a.Shape, grad.Shape);
                var bdims = Shapes.GetBroadcastedAxis(b.Shape, grad.Shape);
                return
                [
                    Sum(grad, axis: adims, scale: beta).Reshape(a.Shape),
                    Sum(grad, axis: bdims, scale: 1f).Reshape(b.Shape)
                ];
            });
    }

    public static CuTensorNode<T> ElementwiseProduct<T>(CuTensorNode<T> a, CuTensorNode<T> b, float scale = 1f) where T : unmanaged, INumberBase<T>
    {
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var context = GetContext(a, b);
        var output = context.Allocate<T>(shape);
        var plan = context.CreateMultiplyPlan<T>(a, b, output);
        return new(
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

    public static CuTensorNode<T> DotProduct<T>(CuTensorNode<T> a, CuTensorNode<T> b, float scale = 1f) where T : unmanaged, INumberBase<T>
    {
        Shapes.EnsureAreEqual(a.Shape, b.Shape);
        var context = GetContext(a, b);
        var output = context.Allocate<T>([]);
        var plan = context.CreateContractionPlan<T>(a, b, output);
        return new(
            output,
            children: [a, b],
            forward: () => plan.Execute(a, b, output, alpha: scale),
            backward: (grad, _) => [grad * b, a * grad]); // TODO: scale!
    }

    public static CuTensorNode<T> MatrixProduct<T>(CuTensorNode<T> a, CuTensorNode<T> b) where T : unmanaged, INumberBase<T>
    {
        var shape = Shapes.BroadcastMatrixProduct(a.Shape, b.Shape); // desired shape
        var context = GetContext(a, b);
        var output = context.Allocate<T>(shape); // true output

        var modA = PadLeft(a);
        var modB = PadRight(b);
        var modShape = Shapes.BroadcastMatrixProduct(modA.Shape, modB.Shape); // padded shape
        var modOutput = output.Reshape(modShape); // padded output
        var plan = context.CreateMatMulPlan<T>(modA, modB, modOutput);

        return new(
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

    public static CuTensorNode<T> Sum<T>(CuTensorNode<T> a) where T : unmanaged, INumberBase<T>
    {
        var context = GetContext(a);
        var output = context.Allocate<T>([]);
        var plan = context.CreateSumPlan<T>(a, output);
        return new(
            output,
            children: [a],
            forward: () => plan.Execute(a, output),
            backward: (grad, _) => [Broadcast(grad, a.Shape)]);
    }
    
    public static CuTensorNode<T> Sum<T>(CuTensorNode<T> a, HashSet<int> axis, float scale = 1f) where T : unmanaged, INumberBase<T>
    {
        var context = GetContext(a);
        var shape = a.Shape.Reduce(axis);
        var output = context.Allocate<T>(shape);
        var plan = context.CreateSumPlan<T>(a, output, axis);
        return new(
            output,
            children: [a],
            forward: () => plan.Execute(a, output, scale),
            backward: (grad, _) => [Broadcast(grad, a.Shape)]); // TODO: Verify!
    }
    
    public static CuTensorNode<T> Broadcast<T>(CuTensorNode<T> a, Shape shape) where T : unmanaged, INumberBase<T>
    {
        if (!a.Shape.CanBroadcastTo(shape))
            throw new InvalidOperationException($"Can't broadcast {a.Shape} to {shape}");

        var context = GetContext(a);
        var output = context.Allocate<T>(shape);
        var axis = Shapes.GetBroadcastedAxis(a.Shape, shape);
        var plan = context.CreateBroadcastPlan<T>(a, output);
        return new(
            output,
            children: [a],
            forward: () => plan.Execute(a, output, gamma: 0),
            backward: (grad, _) => [Sum(grad, axis)]); // TODO: Verify!
    }

    public static CuTensorNode<T> Sigmoid<T>(CuTensorNode<T> a, float scale = 1f) where T : unmanaged, INumberBase<T>
    {
        var context = GetContext(a);
        var output = context.Allocate<T>(a.Shape);
        var one = context.AllocateOne<T>().AsNode();
        var plan = context.CreateUnaryPlan<T>(a, output, cutensorOperator_t.CUTENSOR_OP_SIGMOID);
        return new(
            output,
            children: [a],
            forward: () => plan.Execute(a.Tensor, output, alpha: scale, gamma: 0),
            backward: (grad, self) => [ElementwiseProduct(grad, ElementwiseProduct(self, one - self), scale)]);
    }

    public static CuTensorNode<T> Tanh<T>(CuTensorNode<T> a, float scale = 1f) where T : unmanaged, INumberBase<T>
    {
        var context = GetContext(a);
        var output = context.Allocate<T>(a.Shape);
        var one = context.AllocateOne<T>().AsNode();
        var plan = context.CreateUnaryPlan<T>(a, output, cutensorOperator_t.CUTENSOR_OP_TANH);
        return new(
            output,
            children: [a],
            forward: () => plan.Execute(a, output, alpha: scale, gamma: 0),
            backward: (grad, self) => [ElementwiseProduct(grad, one - Square(self), scale)]);
    }

    public static CuTensorNode<T> Square<T>(CuTensorNode<T> a) where T : unmanaged, INumberBase<T>
    {
        var context = GetContext(a);
        var output = context.Allocate<T>(a.Shape);
        var plan = context.CreateMultiplyPlan<T>(a, a, output);
        return new(
            output,
            children: [a],
            forward: () => plan.Execute(a, a, output),
            backward: (g, _) => [ElementwiseProduct(g, a, scale: 2)]);
    }

    public static CuTensorNode<T> Transpose<T>(CuTensorNode<T> a) where T : unmanaged, INumberBase<T>
    {
        var axis = a.Shape.GetTransposeAxis();
        return Transpose(a, axis);
    }
    
    public static CuTensorNode<T> Transpose<T>(CuTensorNode<T> a, int[] axis) where T : unmanaged, INumberBase<T>
    {
        if (axis.Length != a.Dimensions)
            throw new InvalidOperationException($"Axis {axis.ToText()} is not valid argument for {a.Shape} shape tensor");

        if (!axis.AllElementsAreUnique())
            throw new InvalidOperationException($"Axis {axis.ToText()} does not contain all axes for {a.Shape} shape tensor");

        var shape = a.Shape.Transpose(axis);
        var context = GetContext(a);
        var output = context.Allocate<T>(shape);
        var plan = context.CreatePermutationPlan<T>(a, output, axis);
        return new(
            output,
            children: [a],
            forward: () => plan.Execute(a, output),
            backward: (grad, _) => [Transpose(grad, axis)]); // TODO: Verify!
    }

    private static CuTensorNode<T> PadLeft<T>(CuTensorNode<T> node) 
        where T : unmanaged, INumberBase<T> =>
        node.IsVector
            ? node.Reshape([1, ..node.Shape])
            : node;
    
    private static CuTensorNode<T> PadRight<T>(CuTensorNode<T> node)
        where T : unmanaged, INumberBase<T> =>
        node.IsVector
            ? node.Reshape([..node.Shape, 1])
            : node;
    
    private static CuContext GetContext<T>(
        T operand)
        where T : IHasContext =>
        operand.GetContext();

    private static CuContext GetContext<T>(
        params T[] operands)
        where T : IHasContext =>
        operands
            .Select(c => c.GetContext())
            .Distinct()
            .Single();
}

public partial class CuTensorNode<T> where T : unmanaged, INumberBase<T>
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
            output,
            children: [this],
            forward: () => {},
            backward: (grad, _) => [grad.Reshape(Shape)]);
    }
}