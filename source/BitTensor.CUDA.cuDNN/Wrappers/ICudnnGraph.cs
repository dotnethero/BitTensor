namespace BitTensor.CUDA.Wrappers;

public interface ICudnnGraph : IDisposable
{
    ICudnnPlan GetExecutionPlan();
}