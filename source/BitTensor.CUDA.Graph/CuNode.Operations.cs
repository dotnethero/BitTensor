using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public static partial class CuNode
{
    public static CudaNode<T> Add<T>(CudaNode<T> a, CudaNode<T> b, float beta = 1f, float gamma = 0f) where T : unmanaged, IFloatingPoint<T>
    {
        var context = GetContext(a, b);
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var plan = context.cuTENSOR.CreateAddPlan<T>(a.Shape, b.Shape, shape);
        return new(
            context,
            shape,
            children: [a, b],
            forward: (output) => plan.Execute(a, b, output, beta: beta, gamma: gamma),
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

    public static CudaNode<T> Multiply<T>(CudaNode<T> a, CudaNode<T> b, float scale = 1f) where T : unmanaged, IFloatingPoint<T>
    {
        var context = GetContext(a, b);
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var plan = context.cuTENSOR.CreateMultiplyPlan<T>(a.Shape, b.Shape, shape);
        return new(
            context,
            shape,
            children: [a, b],
            forward: (output) => plan.Execute(a, b, output, alpha: scale),
            backward: (grad, _) =>
            {
                var agrad = Multiply(grad, b, scale);
                var bgrad = Multiply(grad, a, scale);
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
        var plan = context.cuTENSOR.CreateContractionPlan<T>(a.Shape, b.Shape, []);
        return new(
            context,
            [],
            children: [a, b],
            forward: (output) => plan.Execute(a, b, output, alpha: scale),
            backward: (grad, _) => [grad * b, a * grad]); // TODO: scale!
    }

    public static CudaNode<T> MatMul<T>(CudaNode<T> a, CudaNode<T> b) where T : unmanaged, IFloatingPoint<T>
    {
        if (a.IsScalar ||
            b.IsScalar)
            return Multiply(a, b);

        if (a.IsVector &&
            b.IsVector)
            return DotProduct(a, b);

        var context = GetContext(a, b);
        var shape = Shapes.BroadcastMatrixProduct(a.Shape, b.Shape); // desired shape

        var modA = a.IsVector ? a.Reshape([1, ..a.Shape]) : a;
        var modB = b.IsVector ? b.Reshape([..b.Shape, 1]) : b;
        var modShape = Shapes.BroadcastMatrixProduct(modA.Shape, modB.Shape); // padded shape
        var plan = context.cuTENSOR.CreateMatMulPlan<T>(modA.Shape, modB.Shape, modShape);

        return new(
            context,
            shape,
            children: [a, b],
            forward: (output) => plan.Execute(a, b, output),
            backward: (grad, _) =>
            {
                var gpad = grad.Reshape(modShape);
                var da = MatMul(gpad, modB.Transpose());
                var db = MatMul(modA.Transpose(), gpad);
                
                var adims = Shapes.GetBroadcastedAxis(modA.Shape, da.Shape);
                var bdims = Shapes.GetBroadcastedAxis(modB.Shape, db.Shape);
                var agrad = Sum(da, axis: adims).Reshape(a.Shape);
                var bgrad = Sum(db, axis: bdims).Reshape(b.Shape);
                return [agrad, bgrad];
            });
    }

    public static CudaNode<T> Broadcast<T>(CudaNode<T> a, Shape shape, float scale = 1f, float gamma = 0) where T : unmanaged, IFloatingPoint<T>
    {
        if (!a.Shape.CanBroadcastTo(shape))
            throw new InvalidOperationException($"Can't broadcast {a.Shape} to {shape}");

        var context = GetContext(a);
        var axis = Shapes.GetBroadcastedAxis(a.Shape, shape);
        var plan = context.cuTENSOR.CreateBroadcastPlan<T>(a.Shape, shape);
        return new(
            context,
            shape,
            children: [a],
            forward: (output) => plan.Execute(a, output, alpha: scale, gamma: gamma),
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

        var context = GetContext(a);
        var shape = a.Shape.Transpose(axis);
        var inverted = Axis.InvertPermutation(axis);
        var plan = context.cuTENSOR.CreatePermutationPlan<T>(a.Shape, shape, axis);
        return new(
            context,
            shape,
            children: [a],
            forward: (output) => plan.Execute(a, output),
            backward: (grad, _) => [Transpose(grad, inverted)]);
    }
    
    public static CudaNode<T> Softmax<T>(CudaNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var max = Max(a, [^1], keepDims: true);
        var ex = Exp(a - max);
        var sumex = Sum(ex, [^1], keepDims: true);
        return Multiply(ex, Reciprocal(sumex));
    }

    public static CudaNode<float> Gemm(CudaNode<float> a, CudaNode<float> b, CudaNode<float> c)
    {
        var context = GetContext(a, b, c);
        var matmul = MatMul(a, b);
        var shape = Shapes.Broadcast(matmul.Shape, c.Shape);
        var broadcast = context.cuTENSOR.CreateBroadcastPlan<float>(c.Shape, shape);
        return new(
            context,
            shape,
            children: [a, b, c],
            forward: (output) =>
            {
                matmul.Forward!.Invoke(output);
                broadcast.Execute(c, output, gamma: 1f); // add inplace
            },
            backward: (grad, _) =>
            {
                var grads = matmul.Backward!.Invoke(grad, matmul); // matmul gradients don't depend on C and likewise
                var cdims = Shapes.GetBroadcastedAxis(c.Shape, grad.Shape);
                var cgrad = Sum(grad, axis: cdims).Reshape(c.Shape);
                return [..grads, cgrad];
            });
    }

    private static CudaContext GetContext<T>(
        CudaNode<T> operand) 
        where T : unmanaged, IFloatingPoint<T> =>
        operand.Context;

    private static CudaContext GetContext<T>(
        params CudaNode<T>[] operands)
        where T : unmanaged, IFloatingPoint<T> =>
        operands 
            .Select(c => c.Context)
            .Distinct()
            .Single();
}
