using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public static partial class Ops
{
    public static CudaNode<T> Add<T>(CudaNode<T> a, CudaNode<T> b, float alpha = 1f, float beta = 1f) where T : unmanaged, IFloatingPoint<T>
    {
        var context = CudaContext.GetContext(a, b);
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var plan = context.cuTENSOR.CreateAddPlan<T>(a.Shape, b.Shape, shape);
        return new(
            context,
            shape,
            children: [a, b],
            forward: (output) => plan.Execute(a, b, output, alpha, beta, gamma: 0f),
            backward: (grad, _) =>
            {
                var adims = Shapes.GetBroadcastedAxis(a.Shape, grad.Shape);
                var bdims = Shapes.GetBroadcastedAxis(b.Shape, grad.Shape);
                return
                [
                    Sum(grad, axis: adims, scale: alpha).Reshape(a.Shape),
                    Sum(grad, axis: bdims, scale: beta).Reshape(b.Shape)
                ];
            });
    }

    public static CudaNode<T> Multiply<T>(CudaNode<T> a, CudaNode<T> b, float scale = 1f) where T : unmanaged, IFloatingPoint<T>
    {
        var context = CudaContext.GetContext(a, b);
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
}