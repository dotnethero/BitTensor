using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed unsafe class ReLU : AbstractOperation<float>
{
    internal readonly CudaNode<float> Input;
    internal readonly float Alpha;

    public ReLU(CudaNode<float> input, float alpha = 1f) : base(input.Shape, [input])
    {
        Input = input;
        Alpha = alpha;
    }
    
    public override void Execute()
    {
        Kernels.LeakyReLU(Input.Size, Input.Pointer, Tensor.Pointer, Alpha, CuStream.Default);
    }
    
    public override CudaNode<float>[] Propagate(CudaNode<float> gradient)
    {
        return [Ops.ReLU(gradient, Alpha)];
    }
}