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

using static cuTENSOR;
using static cutensorAlgo_t;
using static cutensorDataType_t;
using static cutensorJitMode_t;
using static cutensorOperator_t;
using static cutensorWorksizePreference_t;

using DType = float;
using DTypeView = ArrayView<float>;
using DShapeView = ArrayView<int>;

internal readonly struct CuBackend : ITensorBackend<CuTensor>
{
    public static void ExecuteBroadcast(CuTensor a, CuTensor c)
    {
        a.EnsureHasUpdatedValues();

        var broadcast = c.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DTypeView>(CuKernels.BroadcastScalar);
        broadcast(c.Size, a.ArrayView, c.ArrayView);
    }

    public static void ExecuteNegate(CuTensor a, CuTensor c)
    {
        a.EnsureHasUpdatedValues();

        var negate = c.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DTypeView>(CuKernels.Negate);
        negate(c.Size, a.ArrayView, c.ArrayView);
    }

    public static void ExecuteSum(CuTensor a, CuTensor c)
    {
        a.EnsureHasUpdatedValues();

        var sum = c.Accelerator.LoadStreamKernel<DTypeView, DTypeView>(CuKernels.SumToScalar);
        var config = GetKernelConfig(a);
        sum(config, a.ArrayView, c.ArrayView);
        c.Accelerator.Synchronize();
    }

    public static void ExecuteSum(CuTensor a, HashSet<int> axis, CuTensor c)
    {
        a.EnsureHasUpdatedValues();

        var aStrides = new int[a.Dimensions];
        var cStrides = new int[a.Dimensions];

        var am = 0;
        var cm = 0;
        for (var m = 0; m < a.Dimensions; ++m)
        {
            aStrides[m] = a.Strides[am++];
            cStrides[m] = axis.Contains(m) ? 0 : c.Strides[cm++];
        }

        var acc = c.Accelerator;

        using var aStridesDev = acc.Allocate1D(aStrides.ToArray());
        using var cStridesDev = acc.Allocate1D(cStrides.ToArray());

        var mem = acc.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DType>(CuKernels.Memset);
        var sum = acc.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DShapeView, DTypeView, DShapeView>(CuKernels.Sum);

        mem(c.Size, c.ArrayView, 0f);
        sum(a.Size, a.ArrayView, aStridesDev.View, c.ArrayView, cStridesDev.View);
    }
    
    public static void ExecuteMemset(CuTensor a, DType value)
    {
        var mem = a.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DType>(CuKernels.Memset);
        mem(a.Size, a.ArrayView, value);
    }

    public static void ExecuteAdd(CuTensor a, CuTensor b, CuTensor c)
    {
        a.EnsureHasUpdatedValues();
        b.EnsureHasUpdatedValues();

        var aStrides = new int[c.Dimensions];
        var bStrides = new int[c.Dimensions];
        var cStrides = new int[c.Dimensions];

        for (var m = c.Dimensions - 1; m >= 0; --m)
        {
            aStrides[m] = m < a.Dimensions && a.Shape[m] == c.Shape[m] ? a.Strides[m] : 0;
            bStrides[m] = m < b.Dimensions && b.Shape[m] == c.Shape[m] ? b.Strides[m] : 0;
            cStrides[m] = c.Strides[m];
        }

        var acc = c.Accelerator;
        
        using var aStridesDev = acc.Allocate1D(aStrides);
        using var bStridesDev = acc.Allocate1D(bStrides);
        using var cStridesDev = acc.Allocate1D(cStrides);

        var add = acc.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DShapeView, DTypeView, DShapeView, DTypeView, DShapeView>(CuKernels.Add);

        add(c.Size, a.ArrayView, aStridesDev.View, b.ArrayView, bStridesDev.View, c.ArrayView, cStridesDev.View);
    }
    
    public static void ExecuteAdd(CuTensor a, DType b, CuTensor c)
    {
        a.EnsureHasUpdatedValues();

        var add = c.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DType, DTypeView>(CuKernels.Add);
        add(c.Size, a.ArrayView, b, c.ArrayView);
    }

    public static void ExecuteMultiply(CuTensor a, CuTensor b, CuTensor c)
    {
        a.EnsureHasUpdatedValues();
        b.EnsureHasUpdatedValues();
        
        var aStrides = new int[c.Dimensions];
        var bStrides = new int[c.Dimensions];
        var cStrides = new int[c.Dimensions];

        for (var m = c.Dimensions - 1; m >= 0; --m)
        {
            aStrides[m] = m < a.Dimensions && a.Shape[m] == c.Shape[m] ? a.Strides[m] : 0;
            bStrides[m] = m < b.Dimensions && b.Shape[m] == c.Shape[m] ? b.Strides[m] : 0;
            cStrides[m] = c.Strides[m];
        }

        var acc = c.Accelerator;
        
        using var aStridesDev = acc.Allocate1D(aStrides);
        using var bStridesDev = acc.Allocate1D(bStrides);
        using var cStridesDev = acc.Allocate1D(cStrides);

        var mul = acc.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DShapeView, DTypeView, DShapeView, DTypeView, DShapeView>(CuKernels.Mul);

        mul(c.Size, a.ArrayView, aStridesDev.View, b.ArrayView, bStridesDev.View, c.ArrayView, cStridesDev.View);
    }

    public static void ExecuteMultiply(CuTensor a, DType b, CuTensor c)
    {
        a.EnsureHasUpdatedValues();

        var mul = c.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DType, DTypeView>(CuKernels.Mul);
        mul(c.Size, a.ArrayView, b, c.ArrayView);
    }
    
    [Obsolete]
    public static unsafe void ExecuteMatMulCustomExample(CuTensor a, CuTensor b, CuTensor c)
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
            c.Pointer,
            CUDA_R_32F,
            c.LastDimension,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_ALGO0);
    }

    public static void ExecuteMatMul(CuTensor a, CuTensor b, CuTensor c)
    {
        a.EnsureHasUpdatedValues();
        b.EnsureHasUpdatedValues();

        using var context = new CuBlas(c.Accelerator);

        var strides = Batching.GetBatchStrides(a, b, ..^2);

        var a_batch_size = a.PrevDimension * a.LastDimension;
        var b_batch_size = b.PrevDimension * b.LastDimension;
        var r_batch_size = c.PrevDimension * c.LastDimension;

        foreach (var atom in Batching.GetMatMulBatches(strides, a, b, c))
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
                c.ArrayView.SubView(atom.BatchIndexR * r_batch_size, r_batch_size),
                c.LastDimension);
        }
    }
    
    public static unsafe void Transpose(CuTensor a, int[] axis, CuTensor b)
    {
        cutensorHandle* handle;
        cutensorPlan* plan;
        cutensorPlanPreference* planPreference;
        cutensorTensorDescriptor* aDescriptor;
        cutensorTensorDescriptor* bDescriptor;
        cutensorOperationDescriptor* operationDescriptor;
        cutensorComputeDescriptor computeDescriptor = new();
        cutensorStatus_t status;

        ulong workspaceSizeEstimate;

        var aShape = stackalloc long[a.Dimensions];
        var bShape = stackalloc long[b.Dimensions];
        var aStrides = stackalloc long[a.Dimensions];
        var bStrides = stackalloc long[b.Dimensions];
        var aModes = stackalloc int[a.Dimensions];
        var bModes = stackalloc int[b.Dimensions];

        for (var i = 0; i < a.Dimensions; ++i)
        {
            aShape[i] = a.Shape[i];
            bShape[i] = b.Shape[i];
            aStrides[i] = a.Strides[i];
            bStrides[i] = b.Strides[i];
            aModes[i] = i;
            bModes[i] = axis[i];
        }

        var alpha = 1f;

        status = cutensorCreate(&handle);
        status = cutensorCreateTensorDescriptor(handle, &aDescriptor, (uint) a.Dimensions, aShape, aStrides, CUTENSOR_R_32F, 256u);
        status = cutensorCreateTensorDescriptor(handle, &bDescriptor, (uint) b.Dimensions, bShape, bStrides, CUTENSOR_R_32F, 256u);
        status = cutensorCreatePermutation(handle, &operationDescriptor, aDescriptor, aModes, CUTENSOR_OP_IDENTITY, bDescriptor, bModes, CUTENSOR_COMPUTE_DESC_32F);
        status = cutensorCreatePlanPreference(handle, &planPreference, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_DEFAULT);

        status = cutensorEstimateWorkspaceSize(handle, operationDescriptor, planPreference, CUTENSOR_WORKSPACE_DEFAULT, &workspaceSizeEstimate);

        status = cutensorCreatePlan(handle, &plan, operationDescriptor, planPreference, workspaceSizeEstimate);
        status = cutensorPermute(handle, plan, &alpha, a.Pointer, b.Pointer, (CUstream_st*)0);

        status = cutensorDestroyPlan(plan);
        status = cutensorDestroyPlanPreference(planPreference);
        status = cutensorDestroyOperationDescriptor(operationDescriptor);
        status = cutensorDestroyTensorDescriptor(bDescriptor);
        status = cutensorDestroyTensorDescriptor(aDescriptor);
    }

    private static KernelConfig GetKernelConfig(AbstractTensor a, int groupSize = 128)
    {
        var gridSize = (a.Size + groupSize - 1) / groupSize;
        var config = new KernelConfig(gridSize, groupSize);
        return config;
    }
}
