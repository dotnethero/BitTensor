using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

public sealed class CuTensorPermutationPlan<T> : IDisposable where T : unmanaged, INumberBase<T>
{
    internal readonly CuTensorDescriptor<T> InputDescriptor;
    internal readonly CuTensorDescriptor<T> OutputDescriptor;
    internal readonly CuTensorPermutation<T> Permutation;
    internal readonly CuTensorPlan PermutationPlan;

    internal CuTensorPermutationPlan(CuTensorContext context, AbstractTensor input, AbstractTensor output, ReadOnlySpan<Index> axis)
    {
        var inputModes = new int[input.Dimensions];
        var outputModes = new int[output.Dimensions];

        for (var i = 0; i < input.Dimensions; ++i)
        {
            inputModes[i] = i;
            outputModes[i] = input.Shape.GetOffset(axis[i]);
        }

        InputDescriptor = new(context, input, inputModes);
        OutputDescriptor = new(context, output, outputModes);

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
