using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

/// <summary>
/// X = A⋅B + C
/// </summary>
public sealed class CudnnGemmReluGraph<T> : ICudnnGraph where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudnnTensorDescriptor<T> A;
    internal readonly CudnnTensorDescriptor<T> B;
    internal readonly CudnnTensorDescriptor<T> AB;
    internal readonly CudnnTensorDescriptor<T> C;
    internal readonly CudnnTensorDescriptor<T> Z;
    internal readonly CudnnTensorDescriptor<T> X;

    internal readonly CudnnMatMulOperation<T> Product;
    internal readonly CudnnPointwiseOperation<T> Bias;
    internal readonly CudnnPointwiseOperation<T> Relu;
    internal readonly CudnnGraph Graph;

    public CudnnGemmReluGraph(
        CudnnContext context,
        AbstractTensor<T> a,
        AbstractTensor<T> b,
        AbstractTensor<T> c,
        AbstractTensor<T> x,
        float alpha)
    {
        A = a.CreateDescriptor(dimensions: 3);
        B = b.CreateDescriptor(dimensions: 3);
        C = c.CreateDescriptor(dimensions: 3);
        X = x.CreateDescriptor(dimensions: 3);

        // temp
        AB = Fusion.CreateVirtualDescriptor<T>(x.Shape, dimensions: 3);
        Z = Fusion.CreateVirtualDescriptor<T>(x.Shape, dimensions: 3);

        // operations
        Product = Fusion.MatMul(A, B, AB);
        Bias = Fusion.Add(AB, C, Z);
        Relu = Fusion.ReLU(Z, X, alpha);

        // graph
        Graph = new CudnnGraph(context, [Product, Bias, Relu]);
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