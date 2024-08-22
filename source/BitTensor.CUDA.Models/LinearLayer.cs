using System.Numerics;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public class Flatten<T> : ILayer<T> where T : unmanaged, IFloatingPoint<T>
{
    public CudaContext Context { get; }
    public CudaWeights<T>[] Parameters { get; } = [];

    public Flatten(CudaContext context)
    {
        Context = context;
    }

    public CudaNode<T> Compose(CudaNode<T> input)
    {
        var batch = input.Shape.Extents[0];
        var size = input.Shape.Strides[0];
        return input.Reshape([batch, size]);
    }
}

public class LinearLayer : ILayer<float>
{
    public CudaContext Context { get; }
    public CudaWeights<float>[] Parameters => [Weights, Bias];
    public CudaWeights<float> Weights { get; }
    public CudaWeights<float> Bias { get; }
    public IEpilogue<float>? Epilogue { get; }
    public ActivationFunction<float>? Activation { get; }
    
    public LinearLayer(CudaContext context, int inputs, int outputs, ActivationFunction<float> activation)
    {
        var weights = context.cuRAND.Normal([inputs, outputs]);
        var bias = context.cuRAND.Normal([outputs]);

        Context = context;
        Weights = new CudaWeights<float>(context, weights);
        Bias = new CudaWeights<float>(context, bias);
        Activation = activation;
    }

    public LinearLayer(CudaContext context, int inputs, int outputs, IEpilogue<float>? epilogue = null)
    {
        var weights = context.cuRAND.Normal([inputs, outputs]);
        var bias = context.cuRAND.Normal([outputs]);

        Context = context;
        Weights = new CudaWeights<float>(context, weights);
        Bias = new CudaWeights<float>(context, bias);
        Epilogue = epilogue;
    }
    
    public CudaNode<float> Compose(CudaNode<float> input)
    {
        var z = Ops.Gemm(input, Weights, Bias, Epilogue);
        return Activation is not null
            ? Activation(z)
            : z;
    }
}
