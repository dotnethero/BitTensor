namespace BitTensor.CUDA.Graph;

public partial class CudaNode<T>
{
    public static CudaNode<T> operator +(CudaNode<T> a, CudaNode<T> b) => Ops.Add(a, b, beta: +1);
    public static CudaNode<T> operator -(CudaNode<T> a, CudaNode<T> b) => Ops.Add(a, b, beta: -1);
    public static CudaNode<T> operator *(CudaNode<T> a, CudaNode<T> b) => Ops.Multiply(a, b);
}
