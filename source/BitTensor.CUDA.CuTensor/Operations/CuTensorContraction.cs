using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Operations;

using static cuTENSOR;

internal sealed unsafe class CuTensorContraction<T> : ICuTensorOperation where T : unmanaged, IFloatingPoint<T>
{
    public CuTensorContext Context { get; }
    public cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorContraction(
        CuTensorContext context,
        CuTensorDescriptor<T> a,
        CuTensorDescriptor<T> b,
        CuTensorDescriptor<T> c,
        CuTensorDescriptor<T> d)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreateContraction(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, a.Transformation,
            b.Descriptor, b.Modes, b.Transformation,
            c.Descriptor, c.Modes, c.Transformation, 
            d.Descriptor, d.Modes, Types.GetComputeType<T>());

        Status.EnsureIsSuccess(status);

        Context = context;
        Descriptor = descriptor;
    }

    public CuTensorPlan CreatePlan() => new(this);

    public CuTensorWorkspace CreateWorkspace(CuTensorPlan plan) => new(plan.WorkspaceSize);

    public void Execute(
        CuTensorPlan plan,
        CuTensorWorkspace ws,
        IDeviceArray<T> a,
        IDeviceArray<T> b,
        IDeviceArray<T> c,
        IDeviceArray<T> d,
        float alpha,
        float beta)
    {
        var yy = CuStream.Default;
        var status = cutensorContract(
            Context.Handle, 
            plan.Plan, 
            &alpha, a.Pointer, b.Pointer, 
            &beta,  c.Pointer, d.Pointer, 
            ws.Pointer, 
            ws.Bytes,
            yy);
        
        Status.EnsureIsSuccess(status);
    }
    
    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}