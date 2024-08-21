namespace BitTensor.CUDA.Graph.Nodes;

public abstract partial class AbstractNode<T>
{
    public static AbstractNode<T> operator +(AbstractNode<T> a, AbstractNode<T> b) => new Add<T>(a, b, alpha: 1, beta: +1);
    public static AbstractNode<T> operator -(AbstractNode<T> a, AbstractNode<T> b) => new Add<T>(a, b, alpha: 1, beta: -1);
    public static AbstractNode<T> operator *(AbstractNode<T> a, AbstractNode<T> b) => new Multiply<T>(a, b);
}
