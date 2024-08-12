using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal sealed class CuRandException(curandStatus status) : Exception($"Operation is not completed: {status}");