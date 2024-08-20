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
    
    public override void Execute(CudaTensor<float> output)
    {
        Kernels.LeakyReLU(A.Size, A.Pointer, output.Pointer, Alpha);
    }

    public override AbstractNode<float>[] Propagate(AbstractNode<float> gradient)
    {
        return [new LeakyReLU(gradient, Alpha)];
    }
}