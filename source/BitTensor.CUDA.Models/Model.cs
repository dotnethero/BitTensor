using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public sealed record Compilation(CuTensorNode Loss, CuTensorNode[] Gradients, CuTensorNode Input, CuTensorNode Output, CuTensorNode Desired);

public abstract class Model : ILayer
{
    public static Model Sequential(ILayer[] layers) => new SequentialModel(layers);

    public abstract CuTensorWeight[] Parameters { get; }

    public abstract CuTensorNode Compute(CuTensorNode input);
    
    public Compilation Compile(CuTensorNode input, CuTensorNode desired)
    {
        var output = Compute(input);
        var diff = output - desired;
        var loss = CuTensorNode.DotProduct(diff, diff, scale: 1f);
        var grads = loss.GetGradients();
        var gradients = grads.By(Parameters);
        return new(loss, gradients, input, output, desired);
    }

    public void Fit(Compilation compilation, float lr, int epochs, bool trace = false)
    {
        for (var i = 0; i < epochs; i++)
        {
            compilation.Input.Invalidate();
            compilation.Desired.Invalidate();
           
            if (trace && (epochs < 10 || i % (epochs / 10) == 0))
            {
                var loss = CuGraphDebug.View(compilation.Loss);
                Console.WriteLine($"Loss={loss}");
            }

            ApplyGradients(Parameters, compilation.Gradients, lr);
        }
    }

    public static void ApplyGradients(CuTensorWeight[] variables, CuTensorNode[] gradients, float lr)
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