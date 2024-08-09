using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

public unsafe interface ICuTensorOperation : IDisposable
{
    CuTensorContext Context { get; }
    cutensorOperationDescriptor* Descriptor { get; }
}