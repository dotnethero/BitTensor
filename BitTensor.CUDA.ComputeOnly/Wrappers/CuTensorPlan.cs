using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

using static cuTENSOR;

internal unsafe class CuTensorPlan : IDisposable
{
    internal readonly CuTensorContext Context;
    internal readonly CuTensorBinaryOperation Operation;

    internal readonly cutensorPlanPreference* PlanReference;
    internal readonly cutensorPlan* Plan;

    public CuTensorPlan(CuTensorBinaryOperation operation)
    {
        cutensorPlan* plan;
        cutensorPlanPreference* planPreference;

        ulong workspaceSizeEstimate;

        var preferenceStatus = cutensorCreatePlanPreference(
            operation.Context.Handle, 
            &planPreference, 
            cutensorAlgo_t.CUTENSOR_ALGO_DEFAULT,
            cutensorJitMode_t.CUTENSOR_JIT_MODE_DEFAULT);
        
        if (preferenceStatus != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(preferenceStatus);

        var estimationStatus = cutensorEstimateWorkspaceSize(
            operation.Context.Handle, 
            operation.Descriptor,
            planPreference, 
            cutensorWorksizePreference_t.CUTENSOR_WORKSPACE_DEFAULT, 
            &workspaceSizeEstimate);
        
        if (estimationStatus != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(estimationStatus);

        var planStatus = cutensorCreatePlan(operation.Context.Handle, &plan, operation.Descriptor, planPreference, workspaceSizeEstimate);
        if (planStatus != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(planStatus);

        Context = operation.Context;
        Operation = operation;
        PlanReference = planPreference;
        Plan = plan;
    }

    public void Execute(CuTensor a, CuTensor b, CuTensor c, float alpha = 1f, float gamma = 1f)
    {
        var status = cutensorElementwiseBinaryExecute(Context.Handle, Plan, &alpha, a.Pointer, &gamma, b.Pointer, c.Pointer, (CUstream_st*) 0);
        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);
    }

    public void Dispose()
    {
        cutensorDestroyPlan(Plan);
        cutensorDestroyPlanPreference(PlanReference);
    }
}