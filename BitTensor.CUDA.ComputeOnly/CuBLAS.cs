﻿using BitTensor.CUDA.ComputeOnly.Wrappers;

namespace BitTensor.CUDA.ComputeOnly;

public static class CuBLAS
{
    public static void Scale(CuTensor a, float b, CuTensor r)
    {
        var context = new CublasContext();

        context.Axpy(a, b, r);
    }

    public static void Multiply(CuTensor a, CuTensor b, CuTensor r)
    {
        var context = new CublasContext();

        context.Gemm(a, b, r);
    }
}