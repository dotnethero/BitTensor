using BitTensor.Abstractions;

namespace BitTensor.Core;

public static class Auto
{
    public static T By<T>(this GetGradientsFunction<T> output, T tensor) where T : AbstractTensorNode<T>, ITensor<T>
    {
        return output([tensor]).Single();
    }

    public static T[] By<T>(this GetGradientsFunction<T> output, IEnumerable<T> tensors) where T : AbstractTensorNode<T>, ITensor<T>
    {
        return output(tensors);
    }

    public static GetGradientsFunction<T> Grad<T>(T output) where T : AbstractTensorNode<T>, IAccumulator<T>, IHasAllocator<T>
    {
        var grads = GetGradients<T>(output);
        return vars => vars
            .Select(v => grads.TryGetValue(v, out var grad) ? grad : output.Allocator.Create(0))
            .ToArray();
    }

    public static Dictionary<T, T> GetGradients<T>(T output) where T : AbstractTensorNode<T>, IAccumulator<T>, IHasAllocator<T>
    {
        if (!output.IsScalar)
            throw new InvalidOperationException($"Gradient only defined for scalar-output functions. Output had shape: {output.Shape.Serialize()}");

        var grads = new Dictionary<T, T>(16)
        {
            [output] = output.Allocator.Create(1)
        };

        var nodes = new Stack<T>(16);

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
                    grads[child].Accumulate(grad);
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