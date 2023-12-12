#pragma warning disable CS8500 // This takes the address of, gets the size of, or declares a pointer to a managed type

using System.Runtime.CompilerServices;

namespace BitTensor.Core;

public sealed partial class Tensor
{
    private readonly Lazy<Tensor> transposeLazy;

    internal static long MaxID;
    internal float[] Data;

    public readonly long Id;
    public readonly int Size;
    public readonly int Dimensions;
    public readonly int[] Shape;
    
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
        get => TransposeHint ?? transposeLazy.Value;
    }

    // tensor properties
    public bool IsEmpty
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Size == 0;
    }

    public bool IsScalar
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Dimensions == 0;
    }

    public bool IsVector
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Dimensions == 1;
    }

    public bool IsRow
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Dimensions > 0 && LastDimension == Size;
    }
    
    public bool IsColumn
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Dimensions > 1 && PrevDimension == Size;
    }

    // utility
    public int BatchDimension { get; set; } = 0;

    // computation tree
    internal readonly Tensor[] Children = [];
    internal readonly List<Tensor> Dependents = new(2); // TODO: make custom collection
    internal readonly ForwardFunction? Forward;
    internal readonly BackwardFunction? Backward;
    internal bool Outdated;
    
    // cache
    internal Tensor? TransposeHint;

    // helpers
    internal readonly Tensor A = null!;
    internal readonly Tensor B = null!;
    internal readonly int LastDimension;
    internal readonly int PrevDimension;

    internal Tensor(int[] shape)
    {
        var size = shape.Product();

        Id = Interlocked.Increment(ref MaxID);
        Data = new float[size];
        Size = size;
        Shape = shape;
        Dimensions = shape.Length;

        if (Dimensions > 0)
        {
            LastDimension = Shape[Dimensions - 1];
        }
        if (Dimensions > 1)
        {
            PrevDimension = Shape[Dimensions - 2];
        }

        transposeLazy = new(Transpose);
    }
    
    internal Tensor(int[] shape, float[] values)
    {
        var size = shape.Product();
        if (size != values.Length)
            throw new InvalidOperationException("Can not fit");

        Id = Interlocked.Increment(ref MaxID);
        Data = values;
        Size = size;
        Shape = shape;
        Dimensions = shape.Length;
        
        if (Dimensions > 0)
        {
            LastDimension = Shape[Dimensions - 1];
        }
        if (Dimensions > 1)
        {
            PrevDimension = Shape[Dimensions - 2];
        }

        transposeLazy = new(Transpose);
    }

    internal unsafe Tensor(int[] shape, Tensor[] children, ForwardFunction forward, BackwardFunction backward) : this(shape)
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
            for (var i = count - 1; i >= 0; i--)
            {
                c[i].Dependents.Add(this);
            }
        }
    }

    internal unsafe Tensor(int[] shape, Tensor[] children, ForwardFunction forward, BackwardFunction backward, float[] values) : this(shape, values)
    {
        Children = children;
        Forward = forward;
        Backward = backward;
        Outdated = true; // TODO: not true for real

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

        fixed(Tensor* c = Children)
            for (var i = Children.Length - 1; i >= 0; i--)
            {
                c[i].EnsureHasUpdatedValues();;
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
