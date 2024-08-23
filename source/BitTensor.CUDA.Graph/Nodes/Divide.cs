﻿using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class Divide<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudaNode<T> A;
    internal readonly CudaNode<T> B;
    internal readonly float Scale;
    internal readonly CuTensorTernaryPlan<T> Plan;

    private static Shape GetShape(AbstractTensor a, AbstractTensor b) => Shapes.Broadcast(a.Shape, b.Shape);

    public Divide(CudaNode<T> a, CudaNode<T> b, float scale = 1f) : base(GetShape(a, b), [a, b])
    {
        A = a;
        B = b;
        Scale = scale;
        Plan = Context.cuTENSOR.CreateMultiplyPlan<T>(a.Shape, Operand.Rcp(b.Shape), Shape);
    }

    public override void Execute()
    {
        Plan.Execute(A, B, Tensor, alpha: Scale, beta: 1f, gamma: 0f);
    }

    public override CudaNode<T>[] Propagate(CudaNode<T> gradient)
    {
        var agrad = new Divide<T>(gradient, B, Scale);
        var bgrad = new Multiply<T>(gradient, A / (B * B), -Scale);
        var adims = Shapes.GetBroadcastedAxis(A.Shape, agrad.Shape);
        var bdims = Shapes.GetBroadcastedAxis(B.Shape, bgrad.Shape);
        return
        [
            Ops.Sum(agrad, axis: adims).Reshape(A.Shape),
            Ops.Sum(bgrad, axis: bdims).Reshape(B.Shape)
        ];
    }
}