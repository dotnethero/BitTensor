using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

/// <summary>
/// X = A⋅B + C
/// </summary>
public sealed class CudnnSoftmaxGraph<T> : ICudnnGraph where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudnnTensorDescriptor<T> Input;
    internal readonly CudnnTensorDescriptor<T> Output;
    
    internal readonly CudnnTensorDescriptor<T> ValueMax;
    internal readonly CudnnTensorDescriptor<T> ValueMinusMax;
    internal readonly CudnnTensorDescriptor<T> Exponent;
    internal readonly CudnnTensorDescriptor<T> ExponentSum;

    internal readonly CudnnReductionOperation<T> OpMax;
    internal readonly CudnnPointwiseOperation<T> OpValueMinusMax;
    internal readonly CudnnPointwiseOperation<T> OpExponent;
    internal readonly CudnnReductionOperation<T> OpExponentSum;
    internal readonly CudnnPointwiseOperation<T> OpDivision;
    
    internal readonly CudnnGraph Graph;

    public CudnnSoftmaxGraph(CudnnContext context,
        AbstractTensor<T> a,
        HashSet<Index> axis,
        AbstractTensor<T> z)
    {
        var shape = a.Shape;
        var reduced = a.Shape.Reduce(axis, keepDims: true);

        Input = a.CreateDescriptor();
        Output = z.CreateDescriptor();
        
        // temp
        ValueMax = Fusion.CreateVirtualDescriptor<T>(reduced);
        ValueMinusMax = Fusion.CreateVirtualDescriptor<T>(shape);
        Exponent = Fusion.CreateVirtualDescriptor<T>(shape);
        ExponentSum = Fusion.CreateVirtualDescriptor<T>(reduced);

        // operations
        OpMax = Fusion.Max(Input, ValueMax);
        OpValueMinusMax = Fusion.Subtract(Input, ValueMax, ValueMinusMax);
        OpExponent = Fusion.Exp(ValueMinusMax, Exponent);
        OpExponentSum = Fusion.Sum(Exponent, ExponentSum);
        OpDivision = Fusion.Divide(Exponent, ExponentSum, Output);
        
        // graph
        Graph = new CudnnGraph(context, [OpMax, OpValueMinusMax, OpExponent, OpExponentSum, OpDivision]);
    }

    public ICudnnPlan GetExecutionPlan() => Graph.GetExecutionPlan();

    public void Dispose()
    {
        Graph.Dispose();

        // operations
        OpMax.Dispose();
        OpValueMinusMax.Dispose();
        OpExponent.Dispose();
        OpExponentSum.Dispose();
        OpDivision.Dispose();
        
        // tensors
        Input.Dispose();
        ValueMax.Dispose();
        ValueMinusMax.Dispose();
        Exponent.Dispose();
        ExponentSum.Dispose();
        Output.Dispose();
    }
}