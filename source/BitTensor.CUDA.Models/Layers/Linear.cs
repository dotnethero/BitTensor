using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models.Layers;

public class Linear : ILayer<float>
{
    public CudaContext Context { get; }
    public CudaWeights<float>[] Parameters => [Weights, Bias];
    public CudaWeights<float> Weights { get; }
    public CudaWeights<float> Bias { get; }
    public ActivationFunction<float>? Activation { get; }
    
    public Linear(CudaContext context, int inputs, int outputs, ActivationFunction<float> activation)
    {
        var weights = context.cuRAND.Normal([inputs, outputs]);
        var bias = context.cuRAND.Normal([outputs]);

        Context = context;
        Weights = new CudaWeights<float>(context, weights);
        Bias = new CudaWeights<float>(context, bias);
        Activation = activation;
    }

    public CudaNode<float> Compose(CudaNode<float> input)
    {
        var z = Ops.Gemm(input, Weights, Bias, CudaBackend.cuDNN);
        return Activation is not null
            ? Activation(z)
            : z;
    }
}
