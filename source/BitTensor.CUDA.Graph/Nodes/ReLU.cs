using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph.Nodes;

public sealed unsafe class ReLU : AbstractOperation<float>
{
    internal readonly AbstractNode<float> Input;
    internal readonly float Alpha;

    public ReLU(AbstractNode<float> input, float alpha = 1f) : base(input.Shape, [input])
    {
        Input = input;
        Alpha = alpha;
    }
    
    public override void EnsureHasUpdatedValue()
    {
        Input.EnsureHasUpdatedValue();
        Kernels.LeakyReLU(Input.Size, Input.Pointer, Tensor.Pointer, Alpha);
    }
    
    public override AbstractNode<float>[] Propagate(AbstractNode<float> gradient)
    {
        return [Ops.ReLU(gradient, Alpha)];
    }
}