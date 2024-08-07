using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

internal class CublasException(cublasStatus_t status) : Exception($"Operation is not completed: {status}");