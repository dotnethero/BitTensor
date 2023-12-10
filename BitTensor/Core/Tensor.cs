namespace BitTensor.Core;

public sealed partial class Tensor
{
    internal static long MaxID;

    public long Id { get; }
    public int Size { get; }
    public int Dimensions { get; }
    public int[] Shape { get; }
    public float[] Data { get; }

    // tensor properties
    public bool IsEmpty => Size == 0;
    public bool IsScalar => Dimensions == 0;
    public bool IsVector => Dimensions == 1;
    
    // utility
    public int BatchDimension { get; set; } = 0;

    // computation tree
    internal Tensor[] Children { get; } = [];
    internal HashSet<Tensor> Dependents { get; } = [];
    internal ForwardFunction? Forward { get; }
    internal BackwardFunction? Backward { get; }
    internal bool Outdated { get; private set; }
    
    // cache
    internal Tensor? TransposeHint { get; init; }

    // helpers
    internal Tensor A => Children[0];
    internal Tensor B => Children[1];

    public ReadOnlySpan<float> Values
    {
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

    internal Tensor(int[] shape, Tensor[] children, ForwardFunction forward, BackwardFunction backward) : this(shape)
    {
        Children = children;
        Forward = forward;
        Backward = backward;
        Outdated = true;

        foreach (var child in Children)
        {
            child.Dependents.Add(this);
        }
    }

    internal Tensor(int[] shape, Tensor[] children, ForwardFunction forward, BackwardFunction backward, float[] values) : this(shape, values)
    {
        Children = children;
        Forward = forward;
        Backward = backward;
        Outdated = true; // TODO: not true for real

        foreach (var child in Children)
        {
            child.Dependents.Add(this);
        }
    }

    internal void EnsureHasUpdatedValues()
    {
        if (!Outdated) return;

        foreach (var child in Children)
        {
            child.EnsureHasUpdatedValues();
        }

        Forward?.Invoke(this);
        Outdated = false;
    }

    internal void Invalidate()
    {
        foreach (var dependent in Dependents)
        {
            dependent.Invalidate();
        }

        Outdated = true;
    }

    public override int GetHashCode() => unchecked((int)Id);
}
