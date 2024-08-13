using System.Numerics;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public sealed record Compilation<T>(
    CuNode<T> Loss,
    CuNode<T>[] Gradients,
    CuNode<T> Input,
    CuNode<T> Desired)
    where T : unmanaged, IFloatingPoint<T>;

public static class Model
{
    public static Model<T> Sequential<T>(ILayer<T>[] layers)
        where T : unmanaged, IFloatingPoint<T> => 
        new SequentialModel<T>(layers);
}

public abstract class Model<T> : ILayer<T> where T : unmanaged, IFloatingPoint<T>
{
    public abstract CuWeights<T>[] Parameters { get; }
    public abstract CuNode<T> Compute(CuNode<T> input);
    
    public Compilation<T> Compile(CuNode<T> input, CuNode<T> desired)
    {
        var output = Compute(input);
        var diff = output - desired;
        var loss = CuNode.DotProduct(diff, diff, scale: 1f);
        var grads = loss.GetGradients();
        var gradients = grads.By(Parameters);
        return new(loss, gradients, input, desired);
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
                CuDebug.WriteLine(compilation.Loss);
            }

            ApplyGradients(Parameters, compilation.Gradients, lr);
        }
    }

    public static void ApplyGradients(CuWeights<T>[] variables, CuNode<T>[] gradients, float lr)
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