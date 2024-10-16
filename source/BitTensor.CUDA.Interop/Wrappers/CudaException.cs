using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal sealed class CudaException(cudaError status) : Exception($"Operation is not completed: {status}");
