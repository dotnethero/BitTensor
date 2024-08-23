using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

/// <summary>
/// Contains information about types and operation configuration
/// </summary>
public unsafe interface ICudnnOperator : IDisposable
{
    cudnnBackendDescriptor_t* Descriptor { get; }
}
