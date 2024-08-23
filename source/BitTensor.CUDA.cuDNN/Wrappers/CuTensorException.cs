using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal sealed class CudnnException(cudnnStatus_t status) : Exception($"Operation is not completed: {status}");