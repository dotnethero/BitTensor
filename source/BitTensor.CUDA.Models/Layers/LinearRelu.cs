using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models.Layers;

public class LinearRelu : ILayer<float>
{
    internal readonly float Alpha;

    public CudaContext Context { get; }
    public CudaWeights<float>[] Parameters => [Weights, Bias];
    public CudaWeights<float> Weights { get; }
    public CudaWeights<float> Bias { get; }
    
    public LinearRelu(CudaContext context, int inputs, int outputs, float alpha)
    {
        var weights = context.cuRAND.Normal([inputs, outputs]);
        var bias = context.cuRAND.Normal([outputs]);

        Alpha = alpha;
        Context = context;
        Weights = new CudaWeights<float>(context, weights);
        Bias = new CudaWeights<float>(context, bias);
    }

    public CudaNode<float> Compose(CudaNode<float> input)
    {
        return Ops.GemmRelu(input, Weights, Bias, Alpha);
    }
}