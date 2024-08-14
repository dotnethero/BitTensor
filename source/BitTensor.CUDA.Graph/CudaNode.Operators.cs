using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public partial class CudaNode<T>
{
    public static CudaNode<T> operator +(CudaNode<T> a, CudaNode<T> b) => CuNode.Add(a, b, beta: +1);
    public static CudaNode<T> operator -(CudaNode<T> a, CudaNode<T> b) => CuNode.Add(a, b, beta: -1);
    public static CudaNode<T> operator *(CudaNode<T> a, CudaNode<T> b) => CuNode.Multiply(a, b);
}
