using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

/// <summary>
/// X = A⋅B + C
/// </summary>
public sealed class CudnnGemmSoftmaxGraph<T> : ICudnnGraph where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudnnTensorDescriptor<T> A;
    internal readonly CudnnTensorDescriptor<T> B;
    internal readonly CudnnTensorDescriptor<T> AB;
    internal readonly CudnnTensorDescriptor<T> C;
    internal readonly CudnnTensorDescriptor<T> Z;
    internal readonly CudnnTensorDescriptor<T> Y;
    internal readonly CudnnTensorDescriptor<T> X;

    internal readonly CudnnMatMulOperation<T> Product;
    internal readonly CudnnPointwiseOperation<T> Bias;
    internal readonly CudnnReductionOperation<T> Max;
    internal readonly CudnnPointwiseOperation<T> Softmax;
    internal readonly CudnnGraph Graph;

    public CudnnGemmSoftmaxGraph(
        CudnnContext context,
        AbstractTensor<T> a,
        AbstractTensor<T> b,
        AbstractTensor<T> c,
        AbstractTensor<T> x)
    {
        var reduction = x.Shape.Reduce([^1], keepDims: true);

        A = a.CreateDescriptor();
        B = b.CreateDescriptor();
        C = c.CreateDescriptor();
        X = x.CreateDescriptor();

        // temp
        AB = Fusion.CreateVirtualDescriptor<T>(x.Shape);
        Z = Fusion.CreateVirtualDescriptor<T>(x.Shape);
        Y = Fusion.CreateVirtualDescriptor<T>(x.Shape);

        // operations
        Product = Fusion.MatMul(A, B, AB);
        Bias = Fusion.Add(AB, C, Z);
        Max = Fusion.Max(Z, Y);
        
        // Softmax = Fusion.Softmax(Z, X);B

        // graph
        Graph = new CudnnGraph(context, [Product, Bias, Softmax]);
    }

    public ICudnnPlan GetExecutionPlan() => Graph.GetExecutionPlan();

    public void Dispose()
    {
        Graph.Dispose();
        Product.Dispose();
        Bias.Dispose();
        AB.Dispose();
        X.Dispose();
        A.Dispose();
        B.Dispose();
        C.Dispose();
    }
}