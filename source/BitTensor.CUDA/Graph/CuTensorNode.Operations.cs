using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public partial class CuTensorNode
{
    public static CuTensorNode operator +(CuTensorNode a, CuTensorNode b) => ElementwiseSum(a, b, beta: +1);

    public static CuTensorNode operator -(CuTensorNode a, CuTensorNode b) => ElementwiseSum(a, b, beta: -1);

    public static CuTensorNode operator *(CuTensorNode a, CuTensorNode b)
    {
        if (a.Tensor.IsScalar ||
            b.Tensor.IsScalar)
            return ElementwiseProduct(a, b);

        if (a.Tensor.IsVector && 
            b.Tensor.IsVector)
            return DotProduct(a, b);

        return MatrixProduct(a, b);
    }
    
    public static CuTensorNode ElementwiseSum(CuTensorNode a, CuTensorNode b, float beta = 1f)
    {
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        return new(
            output,
            children: [a, b],
            forward: () => CuBackend.ElementwiseSum(a.Tensor, b.Tensor, output, beta),
            backward: grad =>
            {
                var adims = Shapes.GetBroadcastedAxis(a.Shape, grad.Shape);
                var bdims = Shapes.GetBroadcastedAxis(b.Shape, grad.Shape);
                return
                [
                    CuTensor.Sum(grad, axis: adims, scale: beta).Reshape(a.Shape),
                    CuTensor.Sum(grad, axis: bdims).Reshape(b.Shape)
                ];
            });
    }

    public static CuTensorNode ElementwiseProduct(CuTensorNode a, CuTensorNode b)
    {
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        return new(
            output,
            children: [a, b],
            forward: () => CuBackend.ElementwiseProduct(a.Tensor, b.Tensor, output),
            backward: grad =>
            {
                using var agrad = CuTensor.ElementwiseProduct(grad, b.Tensor);
                using var bgrad = CuTensor.ElementwiseProduct(grad, a.Tensor);
                var adims = Shapes.GetBroadcastedAxis(a.Shape, agrad.Shape);
                var bdims = Shapes.GetBroadcastedAxis(b.Shape, bgrad.Shape);
                return
                [
                    CuTensor.Sum(agrad, axis: adims).Reshape(a.Shape),
                    CuTensor.Sum(bgrad, axis: bdims).Reshape(b.Shape)
                ];
            });
    }

    public static CuTensorNode DotProduct(CuTensorNode a, CuTensorNode b, float scale = 1f)
    {
        Shapes.EnsureAreEqual(a.Shape, b.Shape);
        var output = new CuTensor([]);
        return new(
            output,
            children: [a, b],
            forward: () => CuBackend.DotProduct(a.Tensor, b.Tensor, output, scale),
            backward: grad => [grad * b.Tensor, a.Tensor * grad]); // TODO: scale!
    }

    public static CuTensorNode MatrixProduct(CuTensorNode a, CuTensorNode b)
    {
        var shape = Shapes.BroadcastMatrixProduct(a.Shape, b.Shape); // desired shape
        var output = new CuTensor(shape); // true output

        var modA = PadLeft(a.Tensor);
        var modB = PadRight(b.Tensor);
        var modShape = Shapes.BroadcastMatrixProduct(modA.Shape, modB.Shape); // padded shape
        var modOutput = output.Reshape(modShape); // padded output

        return new(
            output,
            children: [a, b],
            forward: () => CuBackend.MatrixProduct(modA, modB, modOutput),
            backward: g =>
            {
                var gpad = g.Reshape(modShape);

                using var apadT = CuTensor.Transpose(modA);
                using var bpadT = CuTensor.Transpose(modB);

                var da = gpad * bpadT;
                var db = apadT * gpad;
                
                var ax = Shapes.GetBroadcastedAxis(modA.Shape, da.Shape);
                var bx = Shapes.GetBroadcastedAxis(modB.Shape, db.Shape);

                var ag = CuTensor.Sum(da, axis: ax).Reshape(a.Shape);
                var bg = CuTensor.Sum(db, axis: bx).Reshape(b.Shape);

                return [ag, bg];
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

    private static CuTensor PadLeft(CuTensor tensor) =>
        tensor.IsVector
            ? tensor.PadLeft()
            : tensor;
    
    private static CuTensor PadRight(CuTensor tensor) =>
        tensor.IsVector
            ? tensor.PadRight()
            : tensor;
}