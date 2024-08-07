using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA;

public static unsafe class CuBLAS
{
    public static void Add(CuTensor a, CuTensor b, CuTensor r)
    {
        var context = new CublasContext();
        context.Geam(a, b, r);
    }

    public static void Multiply(CuTensor a, CuTensor b, CuTensor r)
    {
        var context = new CublasContext();
        context.Gemm(a, b, r);
    }

    public static void Scale(CuTensor a, float b, CuTensor r)
    {
        var context = new CublasContext();
        context.Axpy(a, b, r);
    }
}