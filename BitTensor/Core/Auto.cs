using System.Numerics.Tensors;

namespace BitTensor.Core;

public static class Updates
{
    public static void ApplyGradients(Tensor[] variables, Tensor[] gradients, float lr)
    {
        for (var i = 0; i < variables.Length; ++i)
        {
            var vars = variables[i].Values;
            var gradient = gradients[i].Values;
            TensorPrimitives.MultiplyAdd(gradient, -lr, vars, variables[i].Data);
            variables[i].Invalidate();
        }
    }
}