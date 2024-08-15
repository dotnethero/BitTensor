using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

public sealed class CuTensorPermutationPlan<T> : ICuTensorPlan where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CuTensorDescriptor<T> A;
    internal readonly CuTensorDescriptor<T> B;
    internal readonly CuTensorPermutation<T> Permutation;
    internal readonly CuTensorPlan PermutationPlan;
    internal bool IsDisposed;

    internal CuTensorPermutationPlan(CuTensorContext context, Operand a, Shape b, ReadOnlySpan<Index> axis)
    {
        var inputModes = new int[a.Shape.Dimensions];
        var outputModes = new int[b.Dimensions];

        for (var i = 0; i < a.Shape.Dimensions; ++i)
        {
            inputModes[i] = i;
            outputModes[i] = a.Shape.GetOffset(axis[i]);
        }

        A = new(context, a, inputModes);
        B = new(context, b, outputModes);
        Permutation = new(context, A, B);
        PermutationPlan = Permutation.CreatePlan();
    }
    
    public void Execute(IDeviceArray<T> a, IDeviceArray<T> b, float alpha = 1f) =>
        Permutation.Execute(PermutationPlan, a, b, alpha);

    public void Dispose()
    {
        if (IsDisposed) return;

        PermutationPlan.Dispose();
        Permutation.Dispose();
        A.Dispose();
        B.Dispose();
        IsDisposed = true;
    }
}
