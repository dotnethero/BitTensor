using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph.Nodes;

internal abstract class AbstractOperation<T> : CudaNode<T>, IDifferentiable<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly Lazy<CudaTensor<T>> TensorGetter;
    internal readonly CudaNode<T>[] Children;

    public sealed override CudaContext Context { get; }
    public sealed override CudaTensor<T> Tensor => TensorGetter.Value;

    protected AbstractOperation(Shape shape, CudaNode<T>[] children) : base(shape)
    {
        Context = CudaContext.GetContext(children);
        Children = children;
        TensorGetter = new(() => Context.Allocate<T>(shape));
        Outdated = true;
        
        foreach (var child in children)
        {
            child.Dependents.Add(this);
        }
    }

    public CudaNode<T>[] GetChildren() => Children;
    
    public override void EnsureHasUpdatedValues()
    {
        if (!Outdated) return;

        foreach (var child in Children)
        {
            child.EnsureHasUpdatedValues();
        }

        Execute();
        Outdated = false;
    }

    public abstract void Execute();

    public abstract CudaNode<T>[] Propagate(CudaNode<T> gradient);
}
