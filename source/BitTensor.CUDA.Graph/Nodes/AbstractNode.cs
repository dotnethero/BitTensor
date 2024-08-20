using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph.Nodes;

public abstract unsafe class AbstractNode<T> : AbstractTensor, IDeviceArray<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudaContext Context;
    internal readonly Lazy<CudaTensor<T>> TensorGetter;

    public CudaTensor<T> Tensor => TensorGetter.Value;
    public T* Pointer => Tensor.Pointer;

    int IDeviceArray.ElementSize => sizeof(T);
    int IDeviceArray.Size => Shape.ArraySize;

    protected AbstractNode(Shape shape, CudaContext context) : base(shape)
    {
        Context = context;
        TensorGetter = new(() => Context.Allocate<T>(shape));
    }

    protected AbstractNode(Shape shape, AbstractNode<T>[] children) : base(shape)
    {
        Context = CudaContext.GetContext(children);
        TensorGetter = new(() => Context.Allocate<T>(shape));
    }
    
    protected AbstractNode(Shape shape, AbstractNode<T> source, Func<CudaTensor<T>, CudaTensor<T>> transformation) : base(shape)
    {
        Context = CudaContext.GetContext(source);
        TensorGetter = new(() => transformation(source.Tensor));
    }

    public abstract void Execute(CudaTensor<T> output);

    public abstract AbstractNode<T>[] Propagate(AbstractNode<T> gradient);

    public AbstractNode<T> Reshape(Shape shape)
    {
        throw new NotImplementedException();
    }

    public AbstractNode<T> Transpose()
    {
        throw new NotImplementedException();
    }
    
    public void CopyToHost(Span<T> destination) => Tensor.CopyToHost(destination);
    public void CopyToDevice(ReadOnlySpan<T> source) => Tensor.CopyToDevice(source);

    public virtual void DisposeResources() {}

    public void Dispose()
    {
        DisposeResources();

        if (TensorGetter.IsValueCreated)
            TensorGetter.Value.Dispose();
    }
}
