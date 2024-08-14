// ReSharper disable ConvertToPrimaryConstructor

using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public delegate CudaNode<float> ActivationFunction(CudaNode<float> node);

public abstract class AbstractLinearLayer : ILayer<float>
{
    public CudaWeights<float> Weights { get; }
    public CudaWeights<float> Bias { get; }
    public CudaWeights<float>[] Parameters => [Weights, Bias];

    protected AbstractLinearLayer(CudaContext context, int inputs, int outputs)
    {
        var weights = context.cuRAND.Normal([inputs, outputs]);
        var bias = context.cuRAND.Normal([outputs]);

        Weights = new CudaWeights<float>(context, weights);
        Bias = new CudaWeights<float>(context, bias);
    }

    public abstract CudaNode<float> Compute(CudaNode<float> input);
}

public class LinearLayer : AbstractLinearLayer
{
    public ActivationFunction Activation { get; }

    public LinearLayer(CudaContext context, int inputs, int outputs, ActivationFunction activation) : base(context, inputs, outputs)
    {
        Activation = activation;
    }
    
    public override CudaNode<float> Compute(CudaNode<float> input)
    {
        var z = Ops.Gemm(input, Weights, Bias);
        var y = Activation(z);
        return y;
    }
}

public class ReluLinearLayer : AbstractLinearLayer
{
    public float Alpha { get; }

    public ReluLinearLayer(CudaContext context, int inputs, int outputs, float alpha) : base(context, inputs, outputs)
    {
        Alpha = alpha;
    }

    public override CudaNode<float> Compute(
        CudaNode<float> input) => 
        Ops.GemmLeakyReLU(input, Weights, Bias, Alpha);
}

