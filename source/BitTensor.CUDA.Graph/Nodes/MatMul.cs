using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

public class MatMul<T> : AbstractNode<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly AbstractNode<T> A;
    internal readonly AbstractNode<T> B;
    internal readonly AbstractNode<T> PaddedA;
    internal readonly AbstractNode<T> PaddedB;
    internal readonly Shape PaddedShape;
    internal readonly CuTensorMatMulPlan<T> Plan;

    private static Shape GetShape(AbstractTensor a, AbstractTensor b) => Shapes.BroadcastMatrixProduct(a.Shape, b.Shape);

    public MatMul(AbstractNode<T> a, AbstractNode<T> b) : base(GetShape(a, b), [a, b])
    {
        A = a;
        B = b;
        PaddedA = a.IsVector ? a.Reshape([1, ..a.Shape]) : a;
        PaddedB = b.IsVector ? b.Reshape([..b.Shape, 1]) : b;
        PaddedShape = Shapes.BroadcastMatrixProduct(PaddedA.Shape, PaddedB.Shape); // padded shape
        Plan = Context.cuTENSOR.CreateMatMulPlan<T>(PaddedA.Shape, PaddedB.Shape, PaddedShape);
    }

    public override void Execute(CudaTensor<T> output)
    {
        Plan.Execute(A, B, output);
    }

    public override AbstractNode<T>[] Propagate(AbstractNode<T> gradient)
    {
        var gpad = gradient.Reshape(PaddedShape);
        var da = new MatMul<T>(gpad, PaddedB.Transpose());
        var db = new MatMul<T>(PaddedA.Transpose(), gpad);
        var adims = Shapes.GetBroadcastedAxis(PaddedA.Shape, da.Shape);
        var bdims = Shapes.GetBroadcastedAxis(PaddedB.Shape, db.Shape);
        var agrad = new Sum<T>(da, axis: adims).Reshape(A.Shape);
        var bgrad = new Sum<T>(db, axis: bdims).Reshape(B.Shape);
        return [agrad, bgrad];
    }

    public override void DisposeResources()
    {
        Plan.Dispose();
    }
}