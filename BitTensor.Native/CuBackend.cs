using BitTensor.Abstractions;
using BitTensor.CUDA.Internals;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA;

using static cuBLAS;

public readonly unsafe struct CuBackend : ITensorBackend<CuTensor>
{
    public static void ExecuteReshape(CuTensor a, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteBroadcast(CuTensor a, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteNegate(CuTensor a, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteSum(CuTensor a, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteSum(CuTensor a, HashSet<int> axes, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteAdd(CuTensor a, CuTensor b, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteAdd(CuTensor a, float b, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteMultiply(CuTensor a, CuTensor b, CuTensor output)
    {
        using var scope = new CublasScope();

        const int incx = 1;
        const int incy = 1;

        var alpha = 1.0f;
        var beta = 0.0f;

        cublasSsbmv_v2(
            scope.Context, 
            cublasFillMode_t.CUBLAS_FILL_MODE_UPPER, 
            a.Size, 
            0, 
            &alpha, 
            a.Handle, 
            1, 
            b.Handle,
            incx,
            &beta,
            output.Handle,
            incy);
    }

    public static void ExecuteMultiply(CuTensor a, float b, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecutePower(CuTensor a, float b, CuTensor output)
    {
        throw new NotImplementedException();
    }
}
