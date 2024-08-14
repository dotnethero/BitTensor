using System.Numerics;

namespace BitTensor.CUDA.Graph;

public class GradientCollection<T> where T : unmanaged, IFloatingPoint<T>
{
    private readonly Dictionary<CudaNode<T>, CudaNode<T>> _gradients = new(16);
    
    public CudaNode<T> this[CudaNode<T> node]
    {
        get => _gradients[node];
        set => _gradients[node] = value;
    }
    
    public CudaNode<T> By(CudaNode<T> variable) => _gradients[variable];

    public CudaNode<T>[] By(IEnumerable<CudaNode<T>> variables) =>
        variables
            .Select(node => _gradients[node])
            .ToArray();

    public bool ContainsKey(CudaNode<T> node) => _gradients.ContainsKey(node);
    
    public void Push(CudaNode<T> node, CudaNode<T> gradient) => _gradients[node] = gradient;
}
