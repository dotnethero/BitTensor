using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal unsafe interface ICuTensorOperation : IDisposable
{
    CuTensorContext Context { get; }
    cutensorOperationDescriptor* Descriptor { get; }
}