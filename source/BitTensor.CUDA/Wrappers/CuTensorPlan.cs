using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;

namespace BitTensor.CUDA.Wrappers;

using static cuTENSOR;

public unsafe class CuTensorPlan : IDisposable
{
    internal readonly CuTensorContext Context;
    internal readonly ICuTensorOperation Operation;

    internal readonly ulong WorkspaceSize;
    internal readonly cutensorPlanPreference* PlanReference;
    internal readonly cutensorPlan* Plan;

    public CuTensorPlan(ICuTensorOperation operation, bool jit = true)
    {
        ArgumentNullException.ThrowIfNull(operation.Descriptor);

        cutensorPlan* plan;
        cutensorPlanPreference* planPreference;

        ulong workspaceSizeEstimate;

        var jitMode = jit
            ? cutensorJitMode_t.CUTENSOR_JIT_MODE_DEFAULT
            : cutensorJitMode_t.CUTENSOR_JIT_MODE_NONE;

        var preferenceStatus = cutensorCreatePlanPreference(
            operation.Context.Handle, 
            &planPreference, 
            cutensorAlgo_t.CUTENSOR_ALGO_DEFAULT,
            jitMode);
        
        Status.EnsureIsSuccess(preferenceStatus);

        var estimationStatus = cutensorEstimateWorkspaceSize(
            operation.Context.Handle, 
            operation.Descriptor,
            planPreference, 
            cutensorWorksizePreference_t.CUTENSOR_WORKSPACE_DEFAULT, 
            &workspaceSizeEstimate);

        Status.EnsureIsSuccess(estimationStatus);

        var planStatus = cutensorCreatePlan(operation.Context.Handle, &plan, operation.Descriptor, planPreference, workspaceSizeEstimate);
        
        Status.EnsureIsSuccess(planStatus);

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