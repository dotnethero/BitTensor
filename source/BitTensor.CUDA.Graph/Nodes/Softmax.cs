using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph.Nodes;

using OpCode = cutensorOperator_t;

internal sealed class Softmax : CudaOperation<float>
{
    internal readonly CudaNode<float> Input;
    internal readonly HashSet<Index> Axis;

    internal readonly CudaTensor<float> Epsilon;
    internal readonly CudaTensor<float> Reduced;
    internal readonly CuTensorReductionPlan<float> MaxPlan;
    internal readonly CuTensorTernaryPlan<float> ValueMinusMaxPlan;
    internal readonly CuTensorTernaryPlan<float> SoftExponentPlan;
    internal readonly CuTensorReductionPlan<float> SoftExponentSumPlan;
    internal readonly CuTensorTernaryPlan<float> SoftmaxPlan;

    public Softmax(CudaNode<float> input, HashSet<Index> axis) : base(input.Shape, [input])
    {
        Input = input;
        Axis = axis;

        Epsilon = Context.Allocate(Shape.Scalar, [1e-6f]);
        Reduced = Context.Allocate<float>(Shape.Reduce(axis, keepDims: true));
        
        MaxPlan = Context.cuTENSOR.CreateReductionPlan<float>(Shape, Reduced.Shape, axis, OpCode.CUTENSOR_OP_MAX, true);
        ValueMinusMaxPlan = Context.cuTENSOR.CreateAddPlan<float>(Shape, Reduced.Shape, Shape);
        SoftExponentPlan = Context.cuTENSOR.CreateAddPlan<float>(Operand.Exp(Shape), Shape.Scalar, Shape);
        SoftExponentSumPlan = Context.cuTENSOR.CreateReductionPlan<float>(
            Shape,
            Reduced.Shape,
            axis: [^1],
            operation: OpCode.CUTENSOR_OP_ADD,
            keepDims: true);

        SoftmaxPlan = Context.cuTENSOR.CreateMultiplyPlan<float>(
            Shape,
            Operand.Rcp(Reduced.Shape),
            Shape);
    }

    public override void Execute()
    {
        MaxPlan.Execute(Input, Reduced); // temp is max
        ValueMinusMaxPlan.Execute(Input, Reduced, Tensor, alpha: 1, beta: -1); // output is diff
        SoftExponentPlan.Execute(Tensor, Epsilon, Tensor); // outputs is exp + eps
        SoftExponentSumPlan.Execute(Tensor, Reduced); // temp is sum of exp
        SoftmaxPlan.Execute(Tensor, Reduced, Tensor);  
    }
    
    public override CudaNode<float>[] Propagate(CudaNode<float> gradient)
    {
        var sum = Ops.Sum(this * gradient, Axis, keepDims: true);
        var result = this * (gradient - sum);
        return [result];
    }

    public override void DisposeResources()
    {
        // temp tensors are disposed by context

        MaxPlan.Dispose();
        ValueMinusMaxPlan.Dispose();
        SoftExponentPlan.Dispose();
        SoftExponentSumPlan.Dispose();
        SoftmaxPlan.Dispose();
    }
}