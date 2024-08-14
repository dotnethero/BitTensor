using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph;

using Ops = cutensorOperator_t;

public static partial class CuNode
{
    public static CudaNode<T> Reciprocal<T>(CudaNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var context = a.Context;
        var shape = a.Shape;
        var plan = context.cuTENSOR.CreateUnaryPlan<T>(shape, shape, Ops.CUTENSOR_OP_RCP);
        return new(
            context,
            shape,
            children: [a],
            forward: (output) => plan.Execute(a, output, gamma: 0),
            backward: (_, self) => [Multiply(self, self, -1)]); // TODO: Simplify
    }

    public static CudaNode<T> Exp<T>(CudaNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var context = a.Context;
        var shape = a.Shape;
        var plan = context.cuTENSOR.CreateUnaryPlan<T>(shape, shape, Ops.CUTENSOR_OP_EXP);
        return new(
            context,
            shape,
            children: [a],
            forward: (output) => plan.Execute(a, output, gamma: 0),
            backward: (grad, self) => [Multiply(grad, self)]);
    }

    public static CudaNode<T> ReLU<T>(CudaNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var context = a.Context;
        var shape = a.Shape;
        var plan = context.cuTENSOR.CreateUnaryPlan<T>(shape, shape, Ops.CUTENSOR_OP_RELU);
        return new(
            context,
            shape,
            children: [a],
            forward: (output) => plan.Execute(a, output, gamma: 0),
            backward: (grad, _) => [ReLU(grad)]);
    }

    public static unsafe CudaNode<float> LeakyReLU(CudaNode<float> a, float alpha)
    {
        var context = a.Context;
        var shape = a.Shape;
        return new(
            context,
            shape,
            children: [a],
            forward: (output) => Kernels.LeakyReLU(a.Size, a.Pointer, output.Pointer, alpha),
            backward: (grad, _) => [LeakyReLU(grad, alpha)]);
    }
}