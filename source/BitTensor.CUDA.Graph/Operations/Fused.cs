using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph;

public static partial class Ops
{

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

    public static CudaNode<T> Gemm<T>(
        CudaNode<T> a,
        CudaNode<T> b,
        CudaNode<T> c,
        IEpilogue<T> epilogue)
        where T : unmanaged, IFloatingPoint<T>
    {
        var context = CudaContext.GetContext(a, b, c);
        var gemm = Gemm(a, b, c);
        var shape = gemm.Shape;

        return new(
            context,
            shape,
            children: [a, b, c],
            forward: (output) =>
            {
                gemm.Forward!.Invoke(output);
                epilogue.Execute(output);
            },
            backward: (grad, _) =>
            {
                var g1 = epilogue.Propagate(grad);
                var g2 = gemm.Backward!.Invoke(g1, gemm);
                return g2;
            });
    }

    public static unsafe CudaNode<float> GemmLeakyReLU(
        CudaNode<float> a,
        CudaNode<float> b,
        CudaNode<float> c,
        float alpha)
    {
        var epilogue = new Epilogue<float>(
            dest => Kernels.LeakyReLU(dest.Size, dest.Pointer, dest.Pointer, alpha),
            grad => Ops.LeakyReLU(grad, alpha));

        return Gemm(a, b, c, epilogue);
    }
}