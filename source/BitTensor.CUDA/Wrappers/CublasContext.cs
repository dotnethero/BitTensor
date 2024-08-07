using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal unsafe class CublasContext : IDisposable
{
    internal readonly cublasContext* Handle;

    public CublasContext()
    {
        cublasContext* handle;

        var status = cuBLAS.cublasCreate_v2(&handle);
        if (status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            throw new CublasException(status);

        Handle = handle;
    }
    
    public void Axpy(CuTensor a, float alpha, CuTensor r)
    {
        cudaRT.cudaMemset(r.Pointer, 0, (uint)r.Size);

        var status = cuBLAS.cublasSaxpy_v2(
            Handle,
            a.Size,
            &alpha,
            a.Pointer,
            incx: 1,
            r.Pointer,
            incy: 1);

        if (status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            throw new CublasException(status);
    }
    
    public void Geam(CuTensor a, CuTensor b, CuTensor r, float alpha = 1f, float beta = 1f)
    {
        var strides = Batching.GetBatchStrides(a, b, ..^2);

        var a_batch_size = a.PrevDimension * a.LastDimension;
        var b_batch_size = b.PrevDimension * b.LastDimension;
        var c_batch_size = r.PrevDimension * r.LastDimension;
        
        foreach (var batch in Batching.GetMatrixBatches(strides, a, b, r))
        {
            var status = cuBLAS.cublasSgeam(
                Handle,
                cublasOperation_t.CUBLAS_OP_N,
                cublasOperation_t.CUBLAS_OP_N,
                b.LastDimension, // b.T rows
                a.LastDimension, // a.T rows
                &alpha,
                b.Pointer + batch.BatchIndexB * b_batch_size,
                b.LastDimension,
                &beta,
                a.Pointer + batch.BatchIndexA * a_batch_size,
                a.LastDimension,
                r.Pointer + batch.BatchIndexR * c_batch_size,
                r.LastDimension);

            if (status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
                throw new CublasException(status);
        }
    }

    public void Gemm(CuTensor a, CuTensor b, CuTensor r, float alpha = 1f, float beta = 0f)
    {
        var strides = Batching.GetBatchStrides(a, b, ..^2);

        var a_batch_size = a.PrevDimension * a.LastDimension;
        var b_batch_size = b.PrevDimension * b.LastDimension;
        var c_batch_size = r.PrevDimension * r.LastDimension;
        
        foreach (var batch in Batching.GetMatrixBatches(strides, a, b, r))
        {
            var status = cuBLAS.cublasSgemm_v2(
                Handle,
                cublasOperation_t.CUBLAS_OP_N,
                cublasOperation_t.CUBLAS_OP_N,
                b.LastDimension, // b.T rows
                a.PrevDimension, // a.T cols
                a.LastDimension, // a.T rows
                &alpha,
                b.Pointer + batch.BatchIndexB * b_batch_size,
                b.LastDimension,
                a.Pointer + batch.BatchIndexA * a_batch_size,
                a.LastDimension,
                &beta,
                r.Pointer + batch.BatchIndexR * c_batch_size,
                r.LastDimension);

            if (status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
                throw new CublasException(status);
        }
    }

    public void Dispose()
    {
        cuBLAS.cublasDestroy_v2(Handle);
    }
}