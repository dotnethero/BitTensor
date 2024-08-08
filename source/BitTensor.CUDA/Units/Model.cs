using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Units;

public sealed record Compilation(CuTensorNode Loss, CuTensorNode Input, CuTensorNode Desired);

public abstract class Model : ILayer
{
    public static Model Sequential(ILayer[] layers) => new SequentialModel(layers);

    public abstract CuTensorNode[] Parameters { get; }

    public abstract CuTensorNode Compute(CuTensorNode input);
    
    public Compilation Compile(CuTensorNode input, CuTensorNode desired)
    {
        var output = Compute(input);
        var diff = output - desired;
        var loss = CuTensorNode.DotProduct(diff, diff, scale: 1f);
        return new(loss, input, desired);
    }

    public void Fit(Compilation compilation, float lr, int epochs, bool trace = false)
    {
        for (var i = 0; i < epochs; i++)
        {
            compilation.Input.Invalidate();
            compilation.Desired.Invalidate();
           
            using var gradients = compilation.Loss.GetGradients();

            if (trace && (epochs < 10 || i % (epochs / 10) == 0))
            {
                CuDebug.WriteLine(compilation.Loss);
            }

            ApplyGradients(Parameters, gradients.By(Parameters), lr);
        }
    }

    public static void ApplyGradients(CuTensorNode[] variables, CuTensor[] gradients, float lr)
    {
        for (var i = 0; i < variables.Length; ++i)
        {
            var gradient = gradients[i];
            var variable = variables[i];
            CuBackend.AddInplace(gradient, variable.Tensor, -lr);
            variable.Invalidate();
        }
    }
}