﻿using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

public sealed unsafe class CuStream
{
    public static CUstream_st* Default = (CUstream_st*)0;
}