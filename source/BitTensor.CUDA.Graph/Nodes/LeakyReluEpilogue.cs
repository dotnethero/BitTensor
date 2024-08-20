using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph.Nodes;

public unsafe class LeakyReluEpilogue : IEpilogue<float>
{
    internal readonly float Alpha;

    public LeakyReluEpilogue(float alpha = 1f)
    {
        Alpha = alpha;
    }
    
    public void Execute(CudaTensor<float> output)
    {
        Kernels.LeakyReLU(output.Size, output.Pointer, output.Pointer, Alpha);
    }

    public AbstractNode<float> GetGradient(AbstractNode<float> gradient)
    {
        return new LeakyReLU(gradient, Alpha);
    }
}