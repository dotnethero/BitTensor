using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Operations;

using static cuTENSOR;

internal sealed unsafe class CuTensorBinaryOperation<T> : ICuTensorOperation where T : unmanaged, IFloatingPoint<T>
{
    public CuTensorContext Context { get; }
    public cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorBinaryOperation(
        CuTensorContext context,
        CuTensorDescriptor<T> a,
        CuTensorDescriptor<T> b,
        CuTensorDescriptor<T> c,
        cutensorOperator_t opAB)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreateElementwiseBinary(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, a.Transformation,
            b.Descriptor, b.Modes, b.Transformation,
            c.Descriptor, c.Modes, opAB,
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
        float alpha,
        float gamma)
    {
        var status = cutensorElementwiseBinaryExecute(Context.Handle, plan.Plan, &alpha, a.Pointer, &gamma, b.Pointer, c.Pointer, CuStream.Default);
        Status.EnsureIsSuccess(status);
    }
    
    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}