using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public partial class CudaNode<T> where T : unmanaged, IFloatingPoint<T>
{
    public GradientCollection<T> GetGradients()
    {
        EnsureHasUpdatedValues();

        var nodes = new Stack<CudaNode<T>>(16);
        var grads = new GradientCollection<T>();
        var one = Context.CreateNode<T>(T.One);

        nodes.Push(this);
        grads.Push(this, one);

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

                Shapes.EnsureAreEqual(child.Shape, grad.Shape);
                if (grads.ContainsKey(child))
                {
                    using var plan = Context.cuTENSOR.CreateAggregationPlan<T>(grad);
                    plan.Execute(grad, grads[child]);
                }
                else
                {
                    nodes.Push(child);
                    grads.Push(child, grad);
                }
            }
        }

        return grads;
    }
}