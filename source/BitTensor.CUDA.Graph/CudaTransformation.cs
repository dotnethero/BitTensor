using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public abstract class CudaTransformation<T> : CudaNode<T>, IDifferentiable<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly Lazy<CudaTensor<T>> TensorGetter;

    public sealed override CudaContext Context { get; }
    public sealed override CudaTensor<T> Tensor => TensorGetter.Value;
    public CudaNode<T> Source { get; }
    
    protected CudaTransformation(Shape shape, CudaNode<T> source, Func<CudaTensor<T>, CudaTensor<T>> transformation) : base(shape)
    {
        Context = CudaContext.GetContext(source);
        TensorGetter = new(() => transformation(source.Tensor));
        Outdated = true;
        Source = source;
        Source.Dependents.Add(this);
    }

    public CudaNode<T>[] GetChildren() => [Source];

    public sealed override void EnsureHasUpdatedValues()
    {
        Source.EnsureHasUpdatedValues();
        Outdated = false;
    }

    public abstract CudaNode<T>[] Propagate(CudaNode<T> gradient);
}