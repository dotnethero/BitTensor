using BitTensor.CUDA.Graph.Nodes;

namespace BitTensor.CUDA.Graph;

public abstract partial class CudaNode<T>
{
    public static CudaNode<T> operator +(CudaNode<T> a, CudaNode<T> b) => new Add<T>(a, b, alpha: 1, beta: +1);
    public static CudaNode<T> operator -(CudaNode<T> a, CudaNode<T> b) => new Add<T>(a, b, alpha: 1, beta: -1);
    public static CudaNode<T> operator *(CudaNode<T> a, CudaNode<T> b) => new Multiply<T>(a, b);
}
