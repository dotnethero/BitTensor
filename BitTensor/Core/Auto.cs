using System.Numerics.Tensors;

namespace BitTensor.Core;

/// <summary>
/// Function that retrieves gradients of expression with respect to specific variables
/// </summary>
/// <param name="variables">Variables for partial gradient</param>
public delegate Tensor[] GetGradientsFunction(IEnumerable<Tensor> variables);

public static class Auto
{
    public static Tensor By(this GetGradientsFunction output, Tensor tensor)
    {
        return output([tensor]).Single();
    }

    public static Tensor[] By(this GetGradientsFunction output, IEnumerable<Tensor> tensors)
    {
        return output(tensors);
    }

    public static GetGradientsFunction Grad(Tensor output) // TODO: make result a class maybe?
    {
        var grads = GetGradients(output);
        return vars => vars
            .Select(v => grads.TryGetValue(v, out var grad) ? grad : Tensor.Zero)
            .ToArray();
    }

    public static void ApplyGradients(Tensor[] variables, Tensor[] gradients, float lr)
    {
        for (var i = 0; i < variables.Length; ++i)
        {
            var gradient = gradients[i].Values;
            TensorPrimitives.MultiplyAdd(gradient, -lr, variables[i].Values, variables[i].Data);
            variables[i].Invalidate();
        }
    }

    public static Dictionary<Tensor, Tensor> GetGradients(Tensor output)
    {
        var grads = new Dictionary<Tensor, Tensor>(16)
        {
            [output] = Tensor.One
        };

        var nodes = new Stack<Tensor>(16);

        nodes.Push(output);

        while (nodes.Count > 0)
        {
            var node = nodes.Pop();
            if (node.Backward is null)
                continue;

            var children = node.Children;
            var childrenGrads = node.Backward(grads[node], node);
            for (var i = 0; i < children.Length; ++i)
            {
                var child = children[i];
                var grad = childrenGrads[i];
                if (grads.ContainsKey(child))
                {
                    grads[child] += grad;
                }
                else
                {
                    grads[child] = grad;
                    nodes.Push(child);
                }
            }
        }

        return grads;
    }
}