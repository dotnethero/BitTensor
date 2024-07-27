namespace BitTensor.Abstractions;

public interface ITensorNode<T> where T : AbstractTensorNode<T>
{
    static abstract T CreateNode(int[] shape, T[] children, AbstractTensorNode<T>.ForwardFunction forward, AbstractTensorNode<T>.BackwardFunction backward);
    static abstract T CreateReshape(int[] shape, T source);
}