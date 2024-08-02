using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

// ReSharper disable NotAccessedVariable
// ReSharper disable JoinDeclarationAndInitializer

namespace BitTensor.CUDA.ComputeOnly;

using static cuBLAS;
using static cuTENSOR;

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

    public static unsafe void Add(CuTensor a, CuTensor b, CuTensor c)
    {
        cutensorHandle* handle;
        cutensorPlan* plan;
        cutensorPlanPreference* planPreference;
        cutensorTensorDescriptor* aDescriptor;
        cutensorTensorDescriptor* bDescriptor;
        cutensorTensorDescriptor* cDescriptor;
        cutensorOperationDescriptor* operationDescriptor;
        cutensorStatus_t status;

        ulong workspaceSizeEstimate;

        var aShape = stackalloc long[a.Dimensions];
        var bShape = stackalloc long[b.Dimensions];
        var cShape = stackalloc long[c.Dimensions];
        var aStrides = stackalloc long[a.Dimensions];
        var bStrides = stackalloc long[b.Dimensions];
        var cStrides = stackalloc long[c.Dimensions];
        var aModes = stackalloc int[a.Dimensions];
        var bModes = stackalloc int[b.Dimensions];
        var cModes = stackalloc int[c.Dimensions];

        for (var i = 0; i < a.Dimensions; ++i)
        {
            aShape[i] = a.Shape[i];
            bShape[i] = b.Shape[i];
            cShape[i] = c.Shape[i];
            aStrides[i] = a.Strides[i];
            bStrides[i] = b.Strides[i];
            cStrides[i] = c.Strides[i];
            aModes[i] = i;
            bModes[i] = i;
            cModes[i] = i;
        }

        var alpha = 1f;
        var gamma = 1f;

        status = cutensorCreate(&handle);
        status = cutensorCreateTensorDescriptor(handle, &aDescriptor, (uint) a.Dimensions, aShape, aStrides, cutensorDataType_t.CUTENSOR_R_32F, 128u);
        status = cutensorCreateTensorDescriptor(handle, &bDescriptor, (uint) b.Dimensions, bShape, bStrides, cutensorDataType_t.CUTENSOR_R_32F, 128u);
        status = cutensorCreateTensorDescriptor(handle, &cDescriptor, (uint) c.Dimensions, cShape, cStrides, cutensorDataType_t.CUTENSOR_R_32F, 128u);
        status = cutensorCreateElementwiseBinary(
            handle, 
            &operationDescriptor,
            aDescriptor, aModes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            bDescriptor, cModes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            cDescriptor, cModes, cutensorOperator_t.CUTENSOR_OP_ADD,
            CUTENSOR_COMPUTE_DESC_32F);

        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            Console.WriteLine(status);

        status = cutensorCreatePlanPreference(handle, &planPreference, 
            cutensorAlgo_t.CUTENSOR_ALGO_DEFAULT, 
            cutensorJitMode_t.CUTENSOR_JIT_MODE_DEFAULT);

        status = cutensorEstimateWorkspaceSize(handle, operationDescriptor, planPreference, cutensorWorksizePreference_t.CUTENSOR_WORKSPACE_DEFAULT, &workspaceSizeEstimate);
        status = cutensorCreatePlan(handle, &plan, operationDescriptor, planPreference, workspaceSizeEstimate);

        // TODO: check scalar

        status = cutensorElementwiseBinaryExecute(handle, plan, &alpha, a.Pointer, &gamma, b.Pointer, c.Pointer, (CUstream_st*) 0);

        status = cutensorDestroyPlan(plan);
        status = cutensorDestroyPlanPreference(planPreference);
        status = cutensorDestroyOperationDescriptor(operationDescriptor);
        status = cutensorDestroyTensorDescriptor(bDescriptor);
        status = cutensorDestroyTensorDescriptor(aDescriptor);
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