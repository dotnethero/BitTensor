#pragma warning disable CS8500 // This takes the address of, gets the size of, or declares a pointer to a managed type

using System.Runtime.CompilerServices;
using BitTensor.Abstractions;

namespace BitTensor.Core;

public sealed partial class Tensor : AbstractTensorNode<Tensor>, IAccumulator<Tensor>
{
    internal float[] Data;
    internal Lazy<Tensor> TransposeLazy;

    public ReadOnlySpan<float> Values
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            EnsureHasUpdatedValues();
            return Data;
        }
    }

    public Tensor T
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => TransposeLazy.Value;
    }

    // utility
    public int BatchDimension = 0;

    internal Tensor(int[] shape, float[]? values = null) : base(shape)
    {
        Data = values ?? new float[Size];
        TransposeLazy = new(Transpose);
    }

    internal Tensor(int[] shape, Tensor[] children, ForwardFunction forward, BackwardFunction backward, float[]? values = null) : base(shape, children, forward, backward)
    {
        Data = values ?? new float[Size];
        TransposeLazy = new(Transpose);
    }
}
