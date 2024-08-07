namespace BitTensor.CUDA.Graph;

public class CuTensorGradients : IDisposable
{
    private readonly Dictionary<CuTensorNode, CuTensor> _gradients = new(16);
    
    public CuTensor this[CuTensorNode node]
    {
        get => _gradients[node];
        set => _gradients[node] = value;
    }
    public bool ContainsKey(CuTensorNode node) => _gradients.ContainsKey(node);

    public void Push(CuTensorNode node, CuTensor gradient) => _gradients[node] = gradient;

    public void Dispose()
    {
        foreach (var gradient in _gradients.Values)
        {
            gradient.Dispose();
        }
    }
}
