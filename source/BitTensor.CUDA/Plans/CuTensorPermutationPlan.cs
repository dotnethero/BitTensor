using BitTensor.Abstractions;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

public sealed class CuTensorPermutationPlan<T> : IDisposable where T : unmanaged
{
    internal readonly CuTensorDescriptor InputDescriptor;
    internal readonly CuTensorDescriptor OutputDescriptor;
    internal readonly CuTensorPermutation<T> Permutation;
    internal readonly CuTensorPlan PermutationPlan;

    internal CuTensorPermutationPlan(CuTensorContext context, AbstractTensor input, AbstractTensor output, ReadOnlySpan<int> axis)
    {
        var inputModes = new int[input.Dimensions];
        var outputModes = new int[output.Dimensions];

        for (var i = 0; i < input.Dimensions; ++i)
        {
            inputModes[i] = i;
            outputModes[i] = axis[i];
        }

        InputDescriptor = context.CreateDescriptor(input, inputModes);
        OutputDescriptor = context.CreateDescriptor(output, outputModes);

        Permutation = new(context, InputDescriptor, OutputDescriptor);
        PermutationPlan = Permutation.CreatePlan();
    }
    
    public void Execute(IDeviceArray<T> input, IDeviceArray<T> output, float alpha = 1f) =>
        Permutation.Execute(
            PermutationPlan,
            input,
            output,
            alpha);

    public void Dispose()
    {
        PermutationPlan.Dispose();
        Permutation.Dispose();
        OutputDescriptor.Dispose();
        InputDescriptor.Dispose();
    }
}
