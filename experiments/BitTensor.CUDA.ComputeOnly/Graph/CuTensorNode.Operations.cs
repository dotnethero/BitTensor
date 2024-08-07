﻿using BitTensor.Abstractions;

namespace BitTensor.CUDA.ComputeOnly.Graph;

public partial class CuTensorNode
{
    public static CuTensorNode operator +(CuTensorNode a, CuTensorNode b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Tensor.Shape, b.Tensor.Shape);
        var output = new CuTensor(shape);
        return new CuTensorNode(
            output,
            children: [a, b],
            forward: () => CuTensor.Add(a.Tensor, b.Tensor, output),
            backward: (grad) =>
            {
                var adims = Shapes.GetBroadcastedAxis(a.Tensor.Shape, output.Shape);
                var bdims = Shapes.GetBroadcastedAxis(b.Tensor.Shape, output.Shape);
                var agrad = CuTensor.Sum(grad, axis: adims);
                var bgrad = CuTensor.Sum(grad, axis: bdims);
                return
                [
                    CuTensor.Reshape(agrad, a.Tensor.Shape), // shares memory with `agrad`
                    CuTensor.Reshape(bgrad, b.Tensor.Shape),
                ];
            });
    }
}