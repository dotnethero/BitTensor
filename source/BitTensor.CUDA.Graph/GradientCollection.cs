using System.Numerics;

namespace BitTensor.CUDA.Graph;

public class GradientCollection<T> where T : unmanaged, IFloatingPoint<T>
{
    private readonly Dictionary<CuNode<T>, CuNode<T>> _gradients = new(16);
    
    public CuNode<T> this[CuNode<T> node]
    {
        get => _gradients[node];
        set => _gradients[node] = value;
    }
    
    public CuNode<T>[] By(IEnumerable<CuNode<T>> variables) =>
        variables
            .Select(node => _gradients[node])
            .ToArray();

    public bool ContainsKey(CuNode<T> node) => _gradients.ContainsKey(node);
    
    public void Push(CuNode<T> node, CuNode<T> gradient) => _gradients[node] = gradient;
}
