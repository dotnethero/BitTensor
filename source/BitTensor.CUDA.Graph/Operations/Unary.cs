using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph;

using OpCode = cutensorOperator_t;

public static partial class Ops
{
    public static CudaNode<T> Reciprocal<T>(CudaNode<T> a, float scale = 1f) where T : unmanaged, IFloatingPoint<T>
    {
        var context = a.Context;
        var shape = a.Shape;
        var plan = context.cuTENSOR.CreateAggregationPlan<T>(Operand.Rcp(shape), shape);
        return new(
            context,
            shape,
            children: [a],
            forward: (output) => plan.Execute(a, output, alpha: scale, gamma: 0),
            backward: (grad, self) => [grad * Multiply(self, self, -scale)]); // TODO: Simplify square
    }

    public static CudaNode<T> Exp<T>(CudaNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var context = a.Context;
        var shape = a.Shape;
        var plan = context.cuTENSOR.CreateAggregationPlan<T>(Operand.Exp(shape), shape);
        return new(
            context,
            shape,
            children: [a],
            forward: (output) => plan.Execute(a, output, gamma: 0),
            backward: (grad, self) => [grad * self]);
    }
    
    public static CudaNode<T> Log<T>(CudaNode<T> a, float scale = 1f) where T : unmanaged, IFloatingPoint<T>
    {
        var context = a.Context;
        var shape = a.Shape;
        var plan = context.cuTENSOR.CreateAggregationPlan<T>(Operand.Log(shape), shape);
        return new(
            context,
            shape,
            children: [a],
            forward: (output) => plan.Execute(a, output, alpha: scale, gamma: 0),
            backward: (grad, self) => [grad * Reciprocal(a, scale)]);
    }

    public static CudaNode<T> ReLU<T>(CudaNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var context = a.Context;
        var shape = a.Shape;
        var plan = context.cuTENSOR.CreateAggregationPlan<T>(Operand.Relu(shape), shape);
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

    public static CudaNode<float> Softmax(CudaNode<float> a) => Softmax(a, axis: [^1]);

    public static CudaNode<float> Softmax(CudaNode<float> a, HashSet<Index> axis)
    {
        var context = a.Context;
        var epsilon = a.Context.CreateNode(1e-6f);
        var shape = a.Shape;
        var reduced = a.Shape.Reduce(axis, keepDims: true);

        var max = context.cuTENSOR.CreateReductionPlan<float>(shape, reduced, axis, OpCode.CUTENSOR_OP_MAX, true);
        var dif = context.cuTENSOR.CreateAddPlan<float>(shape, reduced, shape);

        var eps = context.cuTENSOR.CreateAddPlan<float>(
            Operand.Exp(shape),
            Shape.Scalar,
            shape);

        var smx = context.cuTENSOR.CreateReductionPlan<float>(
            shape,
            reduced,
            axis: [^1],
            operation: OpCode.CUTENSOR_OP_ADD,
            keepDims: true);

        var div = context.cuTENSOR.CreateMultiplyPlan<float>(
            shape,
            Operand.Rcp(reduced),
            shape);

        var temp = context.Allocate<float>(reduced);

        return new(
            context,
            shape,
            children: [a],
            forward: (output) =>
            {
                max.Execute(a, temp); // temp is max
                dif.Execute(a, temp, output, alpha: 1, beta: -1); // output is diff
                eps.Execute(output, epsilon, output); // outputs is exp + eps
                smx.Execute(output, temp); // temp is sum of exp
                div.Execute(output, temp, output);                
            },
            backward: (grad, output) =>
            {
                var product = Sum(Multiply(output, grad), axis, keepDims: true);
                return [output * (grad - product)];
            });
    }
}