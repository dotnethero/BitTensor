using System.Numerics;

namespace BitTensor.CUDA.Graph;

public class CuTensorGradients<T> where T : unmanaged, IFloatingPoint<T>
{
    private readonly Dictionary<CuTensorNode<T>, CuTensorNode<T>> _gradients = new(16);
    
    public CuTensorNode<T> this[CuTensorNode<T> node]
    {
        get => _gradients[node];
        set => _gradients[node] = value;
    }
    
    public CuTensorNode<T>[] By(IEnumerable<CuTensorNode<T>> variables) =>
        variables
            .Select(node => _gradients[node])
            .ToArray();

    public bool ContainsKey(CuTensorNode<T> node) => _gradients.ContainsKey(node);
    
    public void Push(CuTensorNode<T> node, CuTensorNode<T> gradient) => _gradients[node] = gradient;
}
