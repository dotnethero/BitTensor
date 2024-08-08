using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public partial class CuTensorNode
{
    public static CuTensorNode operator +(CuTensorNode a, CuTensorNode b)
    {
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        return new(
            output,
            children: [a, b],
            forward: () => CuBackend.Add(a.Tensor, b.Tensor, output),
            backward: (grad) =>
            {
                var adims = Shapes.GetBroadcastedAxis(a.Shape, output.Shape);
                var bdims = Shapes.GetBroadcastedAxis(b.Shape, output.Shape);
                return
                [
                    CuTensor.Sum(grad, axis: adims).Reshape(a.Shape),
                    CuTensor.Sum(grad, axis: bdims).Reshape(b.Shape)
                ];
            });
    }

    public static CuTensorNode operator *(CuTensorNode a, CuTensorNode b)
    {
        var shape = Shapes.BroadcastMatMul(a.Shape, b.Shape);
        var x = new CuTensor(shape);
        return new(
            x,
            children: [a, b],
            forward: () => CuBackend.Multiply(a.Tensor, b.Tensor, x),
            backward: g =>
            {
                using var aT = CuTensor.Transpose(a.Tensor);
                using var bT = CuTensor.Transpose(b.Tensor);

                var aShape = a.Shape.Dimensions > 2 ? a.Shape[..^2] : [];
                var bShape = b.Shape.Dimensions > 2 ? b.Shape[..^2] : [];
                var xShape = x.Shape.Dimensions > 2 ? x.Shape[..^2] : [];

                var adims = Shapes.GetBroadcastedAxis(aShape, xShape);
                var bdims = Shapes.GetBroadcastedAxis(bShape, xShape);

                var agrad =
                    g.IsVector &&
                    b.Tensor.IsVector
                    ? CuTensor.Outer(g, bT)
                    : CuTensor.Sum(g * bT, axis: adims).Reshape(a.Shape);
                
                var bgrad = 
                    g.IsVector &&
                    a.Tensor.IsVector
                    ? CuTensor.Outer(aT, g)
                    : CuTensor.Sum(aT * g, axis: bdims).Reshape(b.Shape);

                return
                [
                    agrad,
                    bgrad,
                ];
            });
    }

    public static CuTensorNode Sum(CuTensorNode a)
    {
        var output = new CuTensor([]);
        return new(
            output,
            children: [a],
            forward: () => CuBackend.Sum(a.Tensor, output),
            backward: grad => [CuTensor.Broadcast(grad, a.Shape)]);
    }
}