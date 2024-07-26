using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

using static BitTensor.CUDA.Interop.cudaRT;
using static BitTensor.CUDA.Interop.cudaMemcpyKind;
using static BitTensor.CUDA.Interop.cuBLAS;
using static BitTensor.CUDA.Interop.cublasStatus_t;
using static BitTensor.CUDA.Interop.cublasOperation_t;

namespace BitTensor.CUDA;

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
        cublasContext* handle;

        var status = cublasCreate_v2(&handle);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw new InvalidOperationException(status.ToString());
        
        const int incx = 1;
        const int incy = 1;

        var alpha = 1.0f;
        var beta = 0.0f;

        cublasSsbmv_v2(handle, cublasFillMode_t.CUBLAS_FILL_MODE_UPPER, a.Size, 0, &alpha, a.Handle, 1, b.Handle, incx, &beta, output.Handle, incy);
        cublasDestroy_v2(handle);
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
