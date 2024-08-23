using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

/// <summary>
/// Graph node: contains operator and tensor descriptors
/// </summary>
internal unsafe interface ICudnnOperation : IDisposable
{
    cudnnBackendDescriptor_t* Descriptor { get; }
}