using BitTensor.Core;
using BitTensor.Units;
using System.Diagnostics;

namespace BitTensor.Models;

public sealed record Compilation(Tensor Loss, Tensor[] Gradients);

public abstract class Model : ILayer
{
    public static Model Sequential(ILayer[] layers) => new SequentialModel(layers);

    public abstract Tensor[] Parameters { get; }

    public abstract Tensor Compute(Tensor input);
    
    public Compilation Compile(Tensor input, Tensor desired)
    {
        var output = Compute(input);
        var diff = output - desired;
        var loss = Tensor.Sum(diff * diff) * 0.5f;
        var gradients = Auto.Grad(loss).By(Parameters);
        return new(loss, gradients);
    }

    public void Fit(Compilation compilation, float lr, int epochs, bool trace = false)
    {
        for (var i = 0; i < epochs; i++)
        {
            if (trace && i % (epochs / 10) == 0)
            {
                Console.WriteLine(compilation.Loss.Values.Scalar());
            }
            Auto.ApplyGradients(Parameters, compilation.Gradients, lr);
        }
    }
}