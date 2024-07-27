namespace BitTensor.Abstractions;

/// <summary>
/// Declares set of kernels to support backpropagation for operations defined in <see cref="ITensor{T}"/>
/// </summary>
/// <typeparam name="T">Tensor type</typeparam>
public interface ITensorBackend<in T>
{
    static abstract void ExecuteReshape(T a, T output);
    static abstract void ExecuteBroadcast(T a, T output);
    static abstract void ExecuteNegate(T a, T output);
    static abstract void ExecuteSum(T a, T output);
    static abstract void ExecuteSum(T a, HashSet<int> axes, T output);
    static abstract void ExecuteAdd(T a, T b, T output);
    static abstract void ExecuteAdd(T a, float b, T output);
    static abstract void ExecuteMultiply(T a, T b, T output);
    static abstract void ExecuteMultiply(T a, float b, T output);
}