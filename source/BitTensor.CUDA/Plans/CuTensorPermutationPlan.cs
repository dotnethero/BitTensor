using BitTensor.Abstractions;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

internal sealed class CuTensorPermutationPlan : IDisposable
{
    internal readonly CuTensorDescriptor InputDescriptor;
    internal readonly CuTensorDescriptor OutputDescriptor;
    
    internal readonly CuTensorPermutation Permutation;
    internal readonly CuTensorPlan PermutationPlan;

    public CuTensorPermutationPlan(CuTensorContext context, AbstractTensor input, AbstractTensor output, int[] axis)
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

        Permutation = new CuTensorPermutation(context, InputDescriptor, OutputDescriptor);
        PermutationPlan = Permutation.CreatePlan();
    }
    
    public void Execute(CuTensor input, CuTensor output, float alpha = 1f) =>
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
