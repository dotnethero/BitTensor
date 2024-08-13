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
            a.Descriptor, a.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            b.Descriptor, b.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            c.Descriptor, c.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
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
        float alpha = 1f,
        float beta = 1f,
        float gamma = 0f)
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