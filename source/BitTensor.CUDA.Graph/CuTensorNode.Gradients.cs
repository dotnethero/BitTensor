using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph;

public partial class CuTensorNode
{
    public CuTensorGradients GetGradients()
    {
        EnsureHasUpdatedValues();

        var nodes = new Stack<CuTensorNode>(16);
        var grads = new CuTensorGradients();
        var one = Context.AllocateOne().AsNode();

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
                    using var context = new CuTensorContext();
                    using var plan = new CuTensorOffsetPlan(context, grad, grad);
                    plan.Execute(grad.Tensor, grads[child].Tensor, alpha: 1);
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