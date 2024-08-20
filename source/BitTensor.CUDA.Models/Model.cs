using System.Numerics;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public sealed record Compilation<T>(
    CudaNode<T> Loss,
    CudaNode<T>[] Gradients,
    CudaNode<T> Input,
    CudaNode<T> Output,
    CudaNode<T> Desired)
    where T : unmanaged, IFloatingPoint<T>;

public static class Model
{
    public static Model<T> Sequential<T>(ILayer<T>[] layers)
        where T : unmanaged, IFloatingPoint<T> => 
        new SequentialModel<T>(layers);
}

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
        return Ops.Sum(nll, scale: 1); // TODO: scale down by batch size
    }
}

public abstract class Model<T> : ILayer<T> where T : unmanaged, IFloatingPoint<T>
{
    public abstract CudaWeights<T>[] Parameters { get; }
    public abstract CudaNode<T> Compute(CudaNode<T> input);
    
    public Compilation<T> Compile(CudaNode<T> input, CudaNode<T> desired, LossFunction<T> lossFunction)
    {
        var output = Compute(input);
        var loss = lossFunction(output, desired);
        var gradients = loss.GetGradients().By(Parameters);
        return new(loss, gradients, input, output, desired);
    }

    public void Fit(Compilation<T> compilation, float lr, int epochs, bool trace = false)
    {
        for (var i = 0; i < epochs; i++)
        {
            compilation.Input.Invalidate();
            compilation.Desired.Invalidate();
            compilation.Loss.EnsureHasUpdatedValues();
           
            if (trace && (epochs < 10 || i % (epochs / 10) == 0))
            {
                var loss = CuDebug.View(compilation.Loss);
                Console.WriteLine(loss);
            }

            ApplyGradients(Parameters, compilation.Gradients, lr);
        }
    }

    public static void ApplyGradients(CudaWeights<T>[] variables, CudaNode<T>[] gradients, float lr)
    {
        for (var i = 0; i < variables.Length; ++i)
        {
            var gradient = gradients[i];
            var variable = variables[i];
            gradient.EnsureHasUpdatedValues();
            variable.AdjustWeights(gradient.Tensor, lr);
        }
    }
}