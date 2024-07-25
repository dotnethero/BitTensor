#pragma warning disable CS8500 // This takes the address of, gets the size of, or declares a pointer to a managed type

using System.Runtime.CompilerServices;

namespace BitTensor.Core;

public sealed partial class Tensor : AbstractTensorNode<Tensor>
{
    internal IAllocation Allocation;
    internal Lazy<Tensor> TransposeLazy;

    public Span<float> Data
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Allocation.Data;
    }

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

    internal Tensor(int[] shape, IAllocator? allocator = null) : base(shape)
    {
        Allocation = (allocator ?? HostAllocator.Instance).Allocate(shape.Product());
        TransposeLazy = new(Transpose);
    }

    internal Tensor(int[] shape, IAllocation allocation) : base(shape)
    {
        Allocation = allocation;
        TransposeLazy = new(Transpose);
    }

    internal Tensor(int[] shape, Tensor[] children, ForwardFunction forward, BackwardFunction backward, IAllocator? allocator = null) : base(shape, children, forward, backward)
    {
        Allocation = (allocator ?? HostAllocator.Instance).Allocate(shape.Product());
        TransposeLazy = new(Transpose);
    }

    internal Tensor(int[] shape, Tensor[] children, ForwardFunction forward, BackwardFunction backward, IAllocation allocation) : base(shape, children, forward, backward)
    {
        Allocation = allocation;
        TransposeLazy = new(Transpose);
    }
}
