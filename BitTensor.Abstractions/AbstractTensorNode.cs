using System.Runtime.CompilerServices;

namespace BitTensor.Abstractions;

public interface ITensorNode<T> where T : AbstractTensorNode<T>
{
    static abstract T Create(int[] shape, T[] children, AbstractTensorNode<T>.ForwardFunction forward, AbstractTensorNode<T>.BackwardFunction backward);
}

public abstract class AbstractTensorNode<T> : AbstractTensor where T : AbstractTensorNode<T>
{
    /// <summary>
    /// Function that recalculates tensor values based on computation tree
    /// </summary>
    /// <param name="self">Tensor itself</param>
    public delegate void ForwardFunction(T self);

    /// <summary>
    /// Function that transforms tensor gradient to children gradients
    /// </summary>
    /// <param name="grad">Parent gradient</param>
    /// <param name="self">Tensor itself</param>
    public delegate T[] BackwardFunction(T grad, T self);

    // helpers
    protected internal readonly T A = null!;
    protected internal readonly T B = null!;
    protected internal readonly T C = null!;

    // computation tree
    protected internal readonly T[] Children;
    protected internal readonly ForwardFunction? Forward;
    protected internal readonly BackwardFunction? Backward;
    protected internal readonly List<AbstractTensorNode<T>> Dependents = new(3);
    protected internal bool Outdated;

    protected internal AbstractTensorNode(int[] shape) : base(shape)
    {
        Children = [];
    }

    protected internal AbstractTensorNode(int[] shape, T[] children, ForwardFunction forward, BackwardFunction backward) : base(shape)
    {
        Children = children;
        Forward = forward;
        Backward = backward;
        Outdated = true;

        var count = Children.Length;
        if (count > 0)
        {
            A = Children[0];
        }
        if (count > 1)
        {
            B = Children[1];
        }
        if (count > 2)
        {
            C = Children[2];
        }

        foreach (var child in Children)
        {
            child.Dependents.Add(this);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    public void EnsureHasUpdatedValues()
    {
        if (!Outdated) return;

        foreach (var child in Children)
        {
            child.EnsureHasUpdatedValues();
        }

        Forward?.Invoke((T)this);
        Outdated = false;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    public void Invalidate()
    {
        foreach (var child in Dependents)
        {
            child.Invalidate();
        }

        Outdated = true;
    }

    public override int GetHashCode() => unchecked((int)Id);
}