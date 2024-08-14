using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public static partial class Ops
{
    public static CudaNode<T> Broadcast<T>(CudaNode<T> a, Shape shape, float scale = 1f, float gamma = 0) where T : unmanaged, IFloatingPoint<T>
    {
        if (!a.Shape.CanBroadcastTo(shape))
            throw new InvalidOperationException($"Can't broadcast {a.Shape} to {shape}");

        var context = a.Context;
        var axis = Shapes.GetBroadcastedAxis(a.Shape, shape);
        var plan = context.cuTENSOR.CreateBroadcastPlan<T>(a.Shape, shape);
        return new(
            context,
            shape,
            children: [a],
            forward: (output) => plan.Execute(a, output, alpha: scale, gamma: gamma),
            backward: (grad, _) => [Sum(grad, axis, scale: scale)]); // TODO: Verify!
    }

    public static CudaNode<T> DotProduct<T>(CudaNode<T> a, CudaNode<T> b, float scale = 1f) where T : unmanaged, IFloatingPoint<T>
    {
        Shapes.EnsureAreEqual(a.Shape, b.Shape);
        var context = CudaContext.GetContext(a, b);
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

        var context = CudaContext.GetContext(a, b);
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

    public static CudaNode<T> Gemm<T>(CudaNode<T> a, CudaNode<T> b, CudaNode<T> c) where T : unmanaged, IFloatingPoint<T>
    {
        var context = CudaContext.GetContext(a, b, c);
        var matmul = MatMul(a, b);
        var shape = Shapes.Broadcast(matmul.Shape, c.Shape);
        var broadcast = context.cuTENSOR.CreateBroadcastPlan<T>(c.Shape, shape);
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
}
