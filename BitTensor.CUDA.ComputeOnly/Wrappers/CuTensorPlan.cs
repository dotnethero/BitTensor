using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

using static cuTENSOR;

internal unsafe class CuTensorPlan : IDisposable
{
    internal readonly CuTensorContext Context;
    internal readonly ICuTensorOperation Operation;

    internal readonly ulong WorkspaceSize;
    internal readonly cutensorPlanPreference* PlanReference;
    internal readonly cutensorPlan* Plan;

    public CuTensorPlan(ICuTensorOperation operation)
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
        WorkspaceSize = workspaceSizeEstimate;
        PlanReference = planPreference;
        Plan = plan;
    }

    public void Dispose()
    {
        cutensorDestroyPlan(Plan);
        cutensorDestroyPlanPreference(PlanReference);
    }
}