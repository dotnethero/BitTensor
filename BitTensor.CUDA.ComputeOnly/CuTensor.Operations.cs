using System.Diagnostics.CodeAnalysis;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

// ReSharper disable NotAccessedVariable
// ReSharper disable JoinDeclarationAndInitializer

namespace BitTensor.CUDA.ComputeOnly;

using static cuBLAS;

public partial class CuTensor
{
    public static CuTensor operator +(CuTensor a, CuTensor b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        Add(a, b, output);
        return output;
    }

    public static CuTensor operator *(CuTensor a, CuTensor b)
    {
        var batchDimensions = Shapes.EnsureShapesAreCompatible(a.Shape[..^2], b.Shape[..^2]);
        var output = new CuTensor([..batchDimensions, a.PrevDimension, b.LastDimension]);
        Multiply(a, b, output);
        return output;
    }

    // inplace operations

    public static void Add(CuTensor a, CuTensor b, CuTensor output)
    {

    }

    public static unsafe void Multiply(CuTensor a, CuTensor b, CuTensor c)
    {
        cublasContext* context;
        cublasStatus_t status;

        var strides = Batching.GetBatchStrides(a, b, ..^2);

        var a_batch_size = a.PrevDimension * a.LastDimension;
        var b_batch_size = b.PrevDimension * b.LastDimension;
        var c_batch_size = c.PrevDimension * c.LastDimension;
        
        var alpha = 1f;
        var beta = 0f;

        status = cublasCreate_v2(&context);

        foreach (var batch in Batching.GetMatMulBatches(strides, a, b, c))
        {
            status = cublasGemmEx(
                context,
                cublasOperation_t.CUBLAS_OP_N,
                cublasOperation_t.CUBLAS_OP_N,
                b.LastDimension, // b.T rows
                a.PrevDimension, // a.T cols
                a.LastDimension, // a.T rows
                &alpha,
                b.Pointer + batch.BatchIndexB * b_batch_size,
                cudaDataType_t.CUDA_R_32F,
                b.LastDimension,
                a.Pointer + batch.BatchIndexA * a_batch_size,
                cudaDataType_t.CUDA_R_32F,
                a.LastDimension,
                &beta,
                c.Pointer + batch.BatchIndexR * c_batch_size,
                cudaDataType_t.CUDA_R_32F,
                c.LastDimension,
                cublasComputeType_t.CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t.CUBLAS_GEMM_ALGO0);
        }
    }
}