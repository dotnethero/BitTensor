using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public partial class CuTensorNode
{
    public static CuTensorNode operator +(CuTensorNode a, CuTensorNode b)
    {
        var shape = Shapes.Broadcast(a.Tensor.Shape, b.Tensor.Shape);
        var output = new CuTensor(shape);
        return new(
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

    public static CuTensorNode operator *(CuTensorNode a, CuTensorNode b)
    {
        var outputShape = CuTensor.GetMultiplicationShape(a.Tensor, b.Tensor);
        var output = new CuTensor(outputShape);
        return new(
            output,
            children: [a, b],
            forward: () => CuTensor.Multiply(a.Tensor, b.Tensor, output),
            backward: (grad) => throw new NotImplementedException());
    }

    public static CuTensorNode Sum(CuTensorNode a)
    {
        var output = new CuTensor([]);
        return new(
            output,
            children: [a],
            forward: () => CuTensor.Sum(a.Tensor, output),
            backward: (grad) => throw new NotImplementedException());
    }
}