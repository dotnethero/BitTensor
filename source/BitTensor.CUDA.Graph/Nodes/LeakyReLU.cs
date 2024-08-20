using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph.Nodes;

public sealed unsafe class LeakyReLU : AbstractOperation<float>
{
    internal readonly AbstractNode<float> A;
    internal readonly float Alpha;

    public LeakyReLU(AbstractNode<float> a, float alpha = 1f) : base(a.Shape, [a])
    {
        A = a;
        Alpha = alpha;
    }
    
    public override void EnsureHasUpdatedValue()
    {
        A.EnsureHasUpdatedValue();
        Kernels.LeakyReLU(A.Size, A.Pointer, Tensor.Pointer, Alpha);
    }

    public override AbstractNode<float>[] Propagate(AbstractNode<float> gradient)
    {
        return [new LeakyReLU(gradient, Alpha)];
    }
}