using System.Diagnostics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Internals;
using BitTensor.CUDA.Interop;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace BitTensor.CUDA;

using static cudaDataType_t;
using static cuBLAS;
using static cublasOperation_t;
using static cublasComputeType_t;
using static cublasGemmAlgo_t;

using DType = float;
using DTypeView = ArrayView<float>;
using DShapeView = ArrayView<int>;

internal readonly struct CuBackend : ITensorBackend<CuTensor>
{
    public static void ExecuteBroadcast(CuTensor a, CuTensor output)
    {
        var broadcast = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DTypeView>(CuKernels.BroadcastScalar);
        broadcast(output.Size, a.ArrayView, output.ArrayView);
    }

    public static void ExecuteNegate(CuTensor a, CuTensor output)
    {
        var negate = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DTypeView>(CuKernels.Negate);
        negate(output.Size, a.ArrayView, output.ArrayView);
    }

    public static void ExecuteSum(CuTensor a, CuTensor output)
    {
        var sum = output.Accelerator.LoadStreamKernel<DTypeView, DTypeView>(CuKernels.SumToScalar);
        var config = GetKernelConfig(a);
        sum(config, a.ArrayView, output.ArrayView);
        output.Accelerator.Synchronize();
    }

    public static void ExecuteSum(CuTensor a, HashSet<int> axis, CuTensor output)
    {
        var sum = output.Accelerator.LoadStreamKernel<DTypeView, DShapeView, DShapeView, DTypeView>(CuKernels.Sum);

        var shape = a.Shape;
        var dims = shape.Length;
        var old_strides = shape.GetStrides();
        var mod_strides = new int[dims];
        var mod_stride = 1;

        for (var m = 0; m < dims; ++m)
        {
            if (!axis.Contains(m))
            {
                mod_strides[m] = mod_stride;
                mod_stride *= shape[m];
            }
        }

        for (var m = 0; m < dims; ++m)
        {
            if (axis.Contains(m))
            {
                mod_strides[m] = 0;
            }
        }

        using var old_strides_buffer = output.Accelerator.Allocate1D(old_strides.ToArray());
        using var mod_strides_buffer = output.Accelerator.Allocate1D(mod_strides.ToArray());

        var config = GetKernelConfig(a);
        sum(config, a.ArrayView, old_strides_buffer.View, mod_strides_buffer.View, output.ArrayView);
        output.Accelerator.Synchronize();
    }
    
    public static void ExecuteMemset(CuTensor tensor, DType value)
    {
        var add = tensor.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DType>(CuKernels.Memset);
        add(tensor.Size, tensor.ArrayView, value);
    }

    public static void ExecuteAdd(CuTensor a, CuTensor b, CuTensor output)
    {
        var add = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DTypeView, DTypeView>(CuKernels.Add);
        add(output.Size, a.ArrayView, b.ArrayView, output.ArrayView);
    }

    public static void ExecuteAdd(CuTensor a, DType b, CuTensor output)
    {
        var add = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DType, DTypeView>(CuKernels.Add);
        add(output.Size, a.ArrayView, b, output.ArrayView);
    }

    public static void ExecuteMultiply(CuTensor a, CuTensor b, CuTensor output)
    {
        var mul = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DTypeView, DTypeView>(CuKernels.Mul);
        mul(output.Size, a.ArrayView, b.ArrayView, output.ArrayView);
    }

    public static void ExecuteMultiply(CuTensor a, DType b, CuTensor output)
    {
        var mul = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DType, DTypeView>(CuKernels.Mul);
        mul(output.Size, a.ArrayView, b, output.ArrayView);
    }
    
    [Obsolete]
    public static unsafe void ExecuteMatMulCustomExample(CuTensor a, CuTensor b, CuTensor r)
    {
        using var scope = new CublasScope();

        var alpha = 1f;
        var beta = 0f;

        cublasGemmEx(
            scope.Context,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            b.LastDimension, // b.T rows
            a.PrevDimension, // a.T cols
            a.LastDimension, // a.T rows
            &alpha,
            b.Pointer,
            CUDA_R_32F,
            b.LastDimension,
            a.Pointer,
            CUDA_R_32F,
            a.LastDimension,
            &beta,
            r.Pointer,
            CUDA_R_32F,
            r.LastDimension,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_ALGO0);
    }

    public static void ExecuteMatMul(CuTensor a, CuTensor b, CuTensor r)
    {
        a.EnsureHasUpdatedValues();
        b.EnsureHasUpdatedValues();

        using var context = new CuBlas(r.Accelerator);

        var strides = Batching.GetBatchStrides(a, b, ..^2);

        var a_batch_size = a.PrevDimension * a.LastDimension;
        var b_batch_size = b.PrevDimension * b.LastDimension;
        var r_batch_size = r.PrevDimension * r.LastDimension;

        foreach (var atom in Batching.GetMatMulBatches(strides, a, b, r))
        {
            context.Gemm(
                CuBlasOperation.NonTranspose,
                CuBlasOperation.NonTranspose,
                b.LastDimension, // b.T rows
                a.PrevDimension, // a.T cols
                a.LastDimension, // a.T rows
                1f,
                b.ArrayView.SubView(atom.BatchIndexB * b_batch_size, b_batch_size),
                b.LastDimension,
                a.ArrayView.SubView(atom.BatchIndexA * a_batch_size, a_batch_size),
                a.LastDimension,
                0f,
                r.ArrayView.SubView(atom.BatchIndexR * r_batch_size, r_batch_size),
                r.LastDimension);
        }
    }
    
    private static KernelConfig GetKernelConfig(AbstractTensor a, int groupSize = 128)
    {
        var gridSize = (a.Size + groupSize - 1) / groupSize;
        var config = new KernelConfig(gridSize, groupSize);
        return config;
    }
}
