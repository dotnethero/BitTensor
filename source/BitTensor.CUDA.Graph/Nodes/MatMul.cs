using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

public sealed class MatMul<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
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

    public override void EnsureHasUpdatedValue()
    {
        ExecuteInto(Tensor);
    }

    public void ExecuteInto(CudaTensor<T> output)
    {
        A.EnsureHasUpdatedValue();
        B.EnsureHasUpdatedValue();
        Plan.Execute(A, B, output);
    }

    public override AbstractNode<T>[] Propagate(AbstractNode<T> gradient)
    {
        var gpad = gradient.Reshape(PaddedShape);
        var da = Ops.MatMul(gpad, PaddedB.Transpose());
        var db = Ops.MatMul(PaddedA.Transpose(), gpad);
        var adims = Shapes.GetBroadcastedAxis(PaddedA.Shape, da.Shape);
        var bdims = Shapes.GetBroadcastedAxis(PaddedB.Shape, db.Shape);
        var agrad = Ops.Sum(da, axis: adims).Reshape(A.Shape);
        var bgrad = Ops.Sum(db, axis: bdims).Reshape(B.Shape);
        return [agrad, bgrad];
    }

    public override void DisposeResources()
    {
        Plan.Dispose();
    }
}