using System.Runtime.CompilerServices;
using BitTensor.Abstractions;
using BitTensor.CUDA.Internals;
using BitTensor.CUDA.Interop;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.Cuda.API;

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
    
    public static unsafe void ExecuteMatMul(CuTensor a, CuTensor b, CuTensor output)
    {
        using var scope = new CublasScope();

        var alpha = 1f;
        var beta = 1f;

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
            output.Pointer,
            CUDA_R_32F,
            output.LastDimension,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_ALGO0);
    }

    public static void ExecuteMatMulCon(CuTensor a, CuTensor b, CuTensor output)
    {
        using var context = new CuBlas(output.Accelerator);
       
        context.Gemm(
            CuBlasOperation.NonTranspose,
            CuBlasOperation.NonTranspose,
            b.LastDimension, // b.T rows
            a.PrevDimension, // a.T cols
            a.LastDimension, // a.T rows
            1f,
            b.ArrayView,
            b.LastDimension,
            a.ArrayView,
            a.LastDimension,
            0f,
            output.ArrayView,
            output.LastDimension);
    }
    
    private static KernelConfig GetKernelConfig(AbstractTensor a)
    {
        var groupSize = 256;
        var gridSize = (a.Size + groupSize - 1) / groupSize;
        var config = new KernelConfig(gridSize, groupSize);
        return config;
    }
}
