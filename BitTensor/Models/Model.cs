using BitTensor.Core;
using BitTensor.Units;
using System.Diagnostics;

namespace BitTensor.Models;

public sealed record Compilation(Tensor Loss, Tensor[] Gradients, Tensor Inputs, Tensor Desired);

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
        return new(loss, gradients, input, desired);
    }

    public void Fit(Compilation compilation, float lr, int epochs, bool shuffle = true, bool trace = false)
    {
        var batchDim = 0;
        var batchSize = compilation.Inputs.Shape[batchDim];
        var batchIndexes = Enumerable.Range(0, batchSize).ToArray();

        for (var i = 0; i < epochs; i++)
        {
            if (shuffle)
            {
                Random.Shared.Shuffle(batchIndexes);
                compilation.Inputs.Shuffle(batchDim, batchIndexes);
                compilation.Desired.Shuffle(batchDim, batchIndexes);
            }

            if (trace && i % (epochs / 10) == 0)
            {
                Console.WriteLine(compilation.Loss.Values.Scalar());
            }
            Auto.ApplyGradients(Parameters, compilation.Gradients, lr);
        }
    }
}