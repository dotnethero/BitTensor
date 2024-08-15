﻿using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public delegate CudaNode<float> ActivationFunction(CudaNode<float> node);

public class LinearLayer : ILayer<float>
{
    public CudaWeights<float>[] Parameters => [Weights, Bias];
    public CudaWeights<float> Weights { get; }
    public CudaWeights<float> Bias { get; }
    public IEpilogue<float> Epilogue { get; }
    public ActivationFunction? Activation { get; }
    
    public LinearLayer(CudaContext context, int inputs, int outputs, ActivationFunction activation)
    {
        var weights = context.cuRAND.Normal([inputs, outputs]);
        var bias = context.cuRAND.Normal([outputs]);

        Weights = new CudaWeights<float>(context, weights);
        Bias = new CudaWeights<float>(context, bias);
        Epilogue = new Identity();
        Activation = activation;
    }

    public LinearLayer(CudaContext context, int inputs, int outputs, IEpilogue<float> epilogue)
    {
        var weights = context.cuRAND.Normal([inputs, outputs]);
        var bias = context.cuRAND.Normal([outputs]);

        Weights = new CudaWeights<float>(context, weights);
        Bias = new CudaWeights<float>(context, bias);
        Epilogue = epilogue;
    }
    
    public CudaNode<float> Compute(CudaNode<float> input)
    {
        var z = Ops.Gemm(input, Weights, Bias, Epilogue);
        return Activation is not null
            ? Activation(z)
            : z;
    }
}
