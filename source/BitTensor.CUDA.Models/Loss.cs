using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public static class Loss
{
    public static CudaNode<float> MSE(CudaNode<float> output, CudaNode<float> desired)
    {
        var diff = output - desired;
        var loss = Ops.DotProduct(diff, diff, scale: 1f); // TODO: scale down by batch size
        return loss;
    }

    public static CudaNode<float> CrossEntropy(CudaNode<float> softmax, CudaNode<float> desired)
    {
        var probs = Ops.Sum(softmax * desired, axis: [^1], keepDims: true);
        var nll = Ops.Log(probs, scale: -1);
        return Ops.Sum(nll, scale: 1f); // TODO: scale down by batch size
    }
}