using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Operations;

using static cuTENSOR;

internal sealed unsafe class CuTensorTernaryOperation<T> : ICuTensorOperation where T : unmanaged, IFloatingPoint<T>
{
    public CuTensorContext Context { get; }
    public cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorTernaryOperation(
        CuTensorContext context,
        CuTensorDescriptor<T> a,
        CuTensorDescriptor<T> b,
        CuTensorDescriptor<T> c,
        CuTensorDescriptor<T> d,
        cutensorOperator_t opAB,
        cutensorOperator_t opABC)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreateElementwiseTrinary(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, a.Transformation,
            b.Descriptor, b.Modes, b.Transformation,
            c.Descriptor, c.Modes, c.Transformation,
            d.Descriptor, d.Modes, opAB, opABC,
            Types.GetComputeType<T>());

        Status.EnsureIsSuccess(status);

        Context = context;
        Descriptor = descriptor;
    }

    public CuTensorPlan CreatePlan() => new(this);

    public void Execute(
        CuTensorPlan plan,
        IDeviceArray<T> a,
        IDeviceArray<T> b,
        IDeviceArray<T> c,
        IDeviceArray<T> d,
        float alpha,
        float beta,
        float gamma)
    {
        var status = cutensorElementwiseTrinaryExecute(
            Context.Handle,
            plan.Plan, 
            &alpha, a.Pointer, 
            &beta,  b.Pointer, 
            &gamma, c.Pointer, 
            d.Pointer, 
            CuStream.Default);

        Status.EnsureIsSuccess(status);
    }
    
    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}