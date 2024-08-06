using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

internal class CuTensorException(cutensorStatus_t status) : Exception($"Operation is not completed: {status}");