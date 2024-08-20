using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph.Nodes;

public abstract class AbstractOperation<T> : AbstractNode<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly Lazy<CudaTensor<T>> TensorGetter;
    
    public sealed override CudaContext Context { get; }
    public sealed override CudaTensor<T> Tensor => TensorGetter.Value;

    protected AbstractOperation(Shape shape, AbstractNode<T>[] children) : base(shape)
    {
        Context = CudaContext.GetContext(children);
        TensorGetter = new(() => Context.Allocate<T>(shape));
    }

    public abstract AbstractNode<T>[] Propagate(AbstractNode<T> gradient);
}
