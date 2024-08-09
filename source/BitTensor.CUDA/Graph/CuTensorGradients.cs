namespace BitTensor.CUDA.Graph;

public class CuTensorGradients : IDisposable
{
    private readonly Dictionary<CuTensorNode, CuTensorNode> _gradients = new(16);
    
    public CuTensorNode this[CuTensorNode node]
    {
        get => _gradients[node];
        set => _gradients[node] = value;
    }
    
    public CuTensorNode[] By(IEnumerable<CuTensorNode> variables) =>
        variables
            .Select(node => _gradients[node])
            .ToArray();

    public bool ContainsKey(CuTensorNode node) => _gradients.ContainsKey(node);
    
    public void Push(CuTensorNode node, CuTensorNode gradient) => _gradients[node] = gradient;

    public void Dispose()
    {
        foreach (var gradient in _gradients.Values)
        {
            gradient.Dispose();
        }
    }
}
