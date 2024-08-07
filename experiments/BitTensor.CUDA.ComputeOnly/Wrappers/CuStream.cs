using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

internal unsafe class CuStream
{
    public static CUstream_st* Default = (CUstream_st*)0;
}