using System.Numerics;

namespace BitTensor.CUDA.Graph;

public interface IEpilogue<T> where T : unmanaged, IFloatingPoint<T>
{
    void Execute(CudaTensor<T> output);
    CudaNode<T> Propagate(CudaNode<T> gradient);
}

public class Epilogue<T> : IEpilogue<T> where T : unmanaged, IFloatingPoint<T>
{
    public delegate void ForwardFuction(CudaTensor<T> dest);
    public delegate CudaNode<T> BackwardFunction(CudaNode<T> grad);

    internal readonly ForwardFuction Forward;
    internal readonly BackwardFunction Backward;

    public Epilogue(ForwardFuction forward, BackwardFunction backward)
    {
        Forward = forward;
        Backward = backward;
    }

    public void Execute(CudaTensor<T> output) => Forward(output);
    public CudaNode<T> Propagate(CudaNode<T> gradient) => Backward(gradient);
}