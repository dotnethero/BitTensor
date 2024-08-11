using System.Numerics;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public sealed record Compilation<T>(
    CuTensorNode<T> Loss,
    CuTensorNode<T>[] Gradients,
    CuTensorNode<T> Input,
    CuTensorNode<T> Desired)
    where T : unmanaged, INumberBase<T>;

public static class Model
{
    public static Model<T> Sequential<T>(ILayer<T>[] layers)
        where T : unmanaged, INumberBase<T> => 
        new SequentialModel<T>(layers);
}

public abstract class Model<T> : ILayer<T> where T : unmanaged, INumberBase<T>
{
    public abstract CuTensorWeights<T>[] Parameters { get; }
    public abstract CuTensorNode<T> Compute(CuTensorNode<T> input);
    
    public Compilation<T> Compile(CuTensorNode<T> input, CuTensorNode<T> desired)
    {
        var output = Compute(input);
        var diff = output - desired;
        var loss = CuTensorNode.DotProduct(diff, diff, scale: 1f);
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
                var loss = CuDebug.View(compilation.Loss.Tensor);
                Console.WriteLine($"Loss={loss}");
            }

            ApplyGradients(Parameters, compilation.Gradients, lr);
        }
    }

    public static void ApplyGradients(CuTensorWeights<T>[] variables, CuTensorNode<T>[] gradients, float lr)
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