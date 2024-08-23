using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

/// <summary>
/// Graph node: contains operator and tensor descriptors
/// </summary>
public unsafe interface ICudnnOperation : IDisposable
{
    cudnnBackendDescriptor_t* Descriptor { get; }
}