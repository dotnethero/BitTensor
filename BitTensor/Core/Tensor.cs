#pragma warning disable CS8500 // This takes the address of, gets the size of, or declares a pointer to a managed type

using System.Runtime.CompilerServices;

namespace BitTensor.Core;

public sealed partial class Tensor
{
    internal static long MaxID;
    internal IAllocation? Allocation;
    internal Lazy<Tensor> TransposeLazy;

    public readonly long Id;
    public readonly int Size;
    public readonly int Dimensions;
    public readonly int[] Shape;
    public readonly int[] Strides;

    public Span<float> Data
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Allocation!.Data;
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

    // tensor properties
    public readonly bool IsEmpty;
    public readonly bool IsScalar;
    public readonly bool IsVector;
    public readonly bool IsRow;
    public readonly bool IsColumn;
    
    // utility
    public int BatchDimension = 0;

    // helpers
    internal readonly Tensor A = null!;
    internal readonly Tensor B = null!;
    internal readonly int PrevDimension;
    internal readonly int LastDimension;

    // computation tree
    internal readonly Tensor[] Children = [];
    internal readonly List<Tensor> Dependents = new(2); // TODO: make custom collection
    internal readonly ForwardFunction? Forward;
    internal readonly BackwardFunction? Backward;
    internal bool Outdated;

    internal unsafe Tensor(int[] shape, IAllocation? allocation = null)
    {
        var size = shape.Product();

        Id = Interlocked.Increment(ref MaxID);
        Allocation = allocation ?? new HostAllocation(size);
        Size = size;
        Shape = shape;
        Strides = shape.GetStrides();
        Dimensions = shape.Length;
        
        IsEmpty = Size == 0;
        IsScalar = Dimensions == 0;
        IsVector = Dimensions == 1;

        fixed (int* sh = shape)
        {
            if (Dimensions > 0)
            {
                LastDimension = sh[Dimensions - 1];
                IsRow = LastDimension == Size;
            }
            if (Dimensions > 1)
            {
                PrevDimension = Shape[Dimensions - 2];
                IsColumn = PrevDimension == Size;
            }
        }
        
        TransposeLazy = new(Transpose);
    }
    
    internal unsafe Tensor(int[] shape, Tensor[] children, ForwardFunction forward, BackwardFunction backward, IAllocation? allocation = null) : this(shape, allocation)
    {
        Children = children;
        Forward = forward;
        Backward = backward;
        Outdated = true;

        fixed (Tensor* c = Children)
        {
            var count = Children.Length;
            if (count > 0)
            {
                A = c[0];
            }
            if (count > 1)
            {
                B = c[1];
            }
            for (var i = Children.Length - 1; i >= 0; i--)
            {
                c[i].Dependents.Add(this);
            }
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    internal unsafe void EnsureHasUpdatedValues()
    {
        if (!Outdated) return;

        fixed (Tensor* c = Children)
        {
            for (var i = Children.Length - 1; i >= 0; i--)
            {
                c[i].EnsureHasUpdatedValues();
            }
        }

        Forward?.Invoke(this);
        Outdated = false;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    internal void Invalidate()
    {
        for (var i = Dependents.Count - 1; i >= 0; i--)
        {
            Dependents[i].Invalidate();
        }

        Outdated = true;
    }

    public override int GetHashCode() => unchecked((int)Id);
}
