using BitTensor.Abstractions;
using BitTensor.CUDA.Internals;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA;

using static cublasOperation_t;
using static cublasFillMode_t;

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
        using var scope = new CublasScope();

        var alpha = 1.0f;
        var beta = 1.0f;

        cublasSgeam(
            scope.Context,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            a.Size,
            1,
            &alpha,
            a.Handle,
            a.Size,
            &beta,
            b.Handle,
            b.Size,
            output.Handle,
            output.Size);
    }

    public static void ExecuteAdd(CuTensor a, float b, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteMultiply(CuTensor a, CuTensor b, CuTensor output)
    {
        using var scope = new CublasScope();

        var alpha = 1.0f;
        var beta = 0.0f;

        cublasSsbmv_v2(
            scope.Context, 
            CUBLAS_FILL_MODE_UPPER, 
            a.Size, 
            0, 
            &alpha, 
            a.Handle, 
            1, 
            b.Handle,
            1,
            &beta,
            output.Handle,
            1);
    }

    public static void ExecuteMultiply(CuTensor a, float b, CuTensor output)
    {
        using var scope = new CublasScope();

        cublasCopyEx(
            scope.Context,
            a.Size,
            a.Handle,
            cudaDataType_t.CUDA_R_32F,
            1,
            output.Handle,
            cudaDataType_t.CUDA_R_32F,
            1);

        cublasSscal_v2(
            scope.Context,
            a.Size,
            &b,
            output.Handle,
            1);
    }

    public static void ExecutePower(CuTensor a, float b, CuTensor output)
    {
        throw new NotImplementedException();
    }
}
