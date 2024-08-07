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
                var agrad = CuTensor.Sum(grad, axis: adims);
                var bgrad = CuTensor.Sum(grad, axis: bdims);
                return
                [
                    CuTensor.Reshape(agrad, a.Shape), // shares memory with `agrad`
                    CuTensor.Reshape(bgrad, b.Shape),
                ];
            });
    }

    public static CuTensorNode operator *(CuTensorNode a, CuTensorNode b)
    {
        var shape = Shapes.BroadcastMatMul(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        return new(
            output,
            children: [a, b],
            forward: () => CuBackend.Multiply(a.Tensor, b.Tensor, output),
            backward: _ => throw new NotImplementedException());
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