namespace BitTensor.CUDA.Graph;

public partial class CuTensorNode
{
    public CuTensorGradients GetGradients()
    {
        var nodes = new Stack<CuTensorNode>(16);
        var grads = new CuTensorGradients();
        var ones = new CuTensor(Tensor.Shape, Enumerable.Repeat(1f, Tensor.Size).ToArray());

        nodes.Push(this);
        grads.Push(this, ones);

        while (nodes.Count > 0)
        {
            var node = nodes.Pop();
            if (node.Backward is null)
                continue;

            var children = node.Children;
            var childrenGrads = node.Backward(grads[node]);
            for (var i = 0; i < children.Length; ++i)
            {
                var child = children[i];
                var grad = childrenGrads[i];

                if (grads.ContainsKey(child))
                {
                    CuTensor.AddInplace(grad, grads[child]);
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