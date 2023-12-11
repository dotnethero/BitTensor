#pragma warning disable CS8500 // This takes the address of, gets the size of, or declares a pointer to a managed type

using System.Runtime.CompilerServices;

namespace BitTensor.Core;

public sealed partial class Tensor
{
    internal static long MaxID;
    internal float[] Data;

    public readonly long Id;
    public readonly int Size;
    public readonly int Dimensions;
    public readonly int[] Shape;

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

    public unsafe (int z2, int z1) LastTwo
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            fixed (int* sh = Shape)
                return (
                    sh[Dimensions - 2],
                    sh[Dimensions - 1]);
        }
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
    internal unsafe Tensor A
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            fixed(Tensor* c = Children)
                return c[0];
        }
    }

    internal unsafe Tensor B
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            fixed(Tensor* c = Children)
                return c[1];
        }
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
    
    internal Tensor(int[] shape)
    {
        var size = shape.Product();

        Id = Interlocked.Increment(ref MaxID);
        Data = new float[size];
        Size = size;
        Shape = shape;
        Dimensions = shape.Length;
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
    }

    internal unsafe Tensor(int[] shape, Tensor[] children, ForwardFunction forward, BackwardFunction backward) : this(shape)
    {
        Children = children;
        Forward = forward;
        Backward = backward;
        Outdated = true;

        fixed(Tensor* c = Children)
            for (var i = Children.Length - 1; i >= 0; i--)
            {
                c[i].Dependents.Add(this);
            }
    }

    internal unsafe Tensor(int[] shape, Tensor[] children, ForwardFunction forward, BackwardFunction backward, float[] values) : this(shape, values)
    {
        Children = children;
        Forward = forward;
        Backward = backward;
        Outdated = true; // TODO: not true for real
        
        fixed(Tensor* c = Children)
            for (var i = Children.Length - 1; i >= 0; i--)
            {
                c[i].Dependents.Add(this);
            }
    }

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
