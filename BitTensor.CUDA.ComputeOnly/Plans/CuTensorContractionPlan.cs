using BitTensor.CUDA.ComputeOnly.Wrappers;

namespace BitTensor.CUDA.ComputeOnly.Plans;

internal sealed class CuTensorContractionPlan : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;
    internal readonly CuTensorDescriptor ResultDescriptor;
    
    internal readonly CuTensorContraction Contraction;
    internal readonly CuTensorPlan ContractionPlan;
    internal readonly CuTensorWorkspace Workspace;
    
    public CuTensorContractionPlan(CuTensorContext context, CuTensor left, CuTensor right, CuTensor result)
    {
        LeftDescriptor = context.CreateDescriptor(left);
        RightDescriptor = context.CreateDescriptor(right);
        ResultDescriptor = context.CreateDescriptor(result);
        
        Contraction = context.CreateContraction(LeftDescriptor, RightDescriptor, ResultDescriptor, ResultDescriptor);
        ContractionPlan = Contraction.CreatePlan();
        Workspace = Contraction.CreateWorkspace(ContractionPlan);
    }
    
    public void Execute(CuTensor left, CuTensor right, CuTensor result)
    {
        Contraction.ExecuteWithPlan(
            ContractionPlan,
            Workspace,
            left,
            right,
            result,
            result,
            alpha: 1,
            beta: 0);
    }
    
    public void Dispose()
    {
        ContractionPlan.Dispose();
        Workspace.Dispose();
        Contraction.Dispose();
        ResultDescriptor.Dispose();
        RightDescriptor.Dispose();
        LeftDescriptor.Dispose();
    }
}