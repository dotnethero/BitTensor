using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

using static cuTENSOR;

internal unsafe class CuTensorBinaryOperation : IDisposable
{
    internal readonly CuTensorContext Context;
    internal readonly CuTensorDescriptor A;
    internal readonly CuTensorDescriptor B;
    internal readonly CuTensorDescriptor C;

    internal readonly cutensorOperationDescriptor* Descriptor;

    public CuTensorBinaryOperation(CuTensorContext context, CuTensorDescriptor a, CuTensorDescriptor b, CuTensorDescriptor c, cutensorOperator_t operation)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreateElementwiseBinary(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            b.Descriptor, c.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            c.Descriptor, c.Modes, operation,
            CUTENSOR_COMPUTE_DESC_32F);

        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);

        A = a;
        B = b;
        C = c;
        Context = context;
        Descriptor = descriptor;
    }

    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}