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
        if (a.Tensor.IsVector &&
            b.Tensor.IsVector)
            return DotProduct(a, b);

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

                using var da = b.Tensor.IsVector ? CuTensor.Outer(g, bT) : g * bT;
                using var db = a.Tensor.IsVector ? CuTensor.Outer(aT, g) : aT * g;
                
                var ax = Shapes.GetBroadcastedAxis(a.Shape, da.Shape);
                var bx = Shapes.GetBroadcastedAxis(b.Shape, db.Shape);

                var ag = CuTensor.Sum(da, axis: ax).Reshape(a.Shape);
                var bg = CuTensor.Sum(db, axis: bx).Reshape(b.Shape);

                return [ag, bg];
            });
    }

    private static CuTensorNode DotProduct(CuTensorNode a, CuTensorNode b)
    {
        var output = new CuTensor([]);
        return new(
            output,
            children: [a, b],
            forward: () => CuBackend.Multiply(a.Tensor, b.Tensor, output),
            backward: grad =>
            [
                CuTensor.Mul(grad, b.Tensor), 
                CuTensor.Mul(a.Tensor, grad), 
            ]);
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