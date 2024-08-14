using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public partial class CudaNode<T>
{
    public static CudaNode<T> operator +(CudaNode<T> a, CudaNode<T> b) => CuNode.Add(a, b, beta: +1);
    public static CudaNode<T> operator -(CudaNode<T> a, CudaNode<T> b) => CuNode.Add(a, b, beta: -1);
    public static CudaNode<T> operator *(CudaNode<T> a, CudaNode<T> b) => CuNode.Multiply(a, b);

    public CudaNode<T> Reshape(Shape shape) =>
        new(Context,
            Shape.EnsureCanReshape(shape),
            children: [this],
            forward: _ => { },
            backward: (grad, _) => [grad.Reshape(Shape)],
            tensor: new(() => Tensor.Reshape(shape)));

    public CudaNode<T> Transpose(Index[] axis) =>
        new(Context,
            Shape.Transpose(axis),
            children: [this],
            forward: _ => { },
            backward: (grad, _) => [grad.Transpose(axis)], // TODO: Verify
            tensor: new(() => Tensor.Transpose(axis)));

    public CudaNode<T> Transpose() => Transpose(Shape.GetTransposeAxis());
}
