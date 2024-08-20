using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph.Nodes;

public abstract class AbstractTransformation<T> : AbstractNode<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly Lazy<CudaTensor<T>> TensorGetter;

    public sealed override CudaContext Context { get; }
    public sealed override CudaTensor<T> Tensor => TensorGetter.Value;
    public AbstractNode<T> Source { get; }
    
    protected AbstractTransformation(Shape shape, AbstractNode<T> source, Func<CudaTensor<T>, CudaTensor<T>> transformation) : base(shape)
    {
        Context = CudaContext.GetContext(source);
        TensorGetter = new(() => transformation(source.Tensor));
        Source = source;
        Source.Dependents.Add(this);
    }

    public abstract AbstractNode<T>[] Propagate(AbstractNode<T> gradient);
}