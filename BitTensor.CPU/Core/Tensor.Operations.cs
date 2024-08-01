using BitTensor.Abstractions;

namespace BitTensor.Core;

using Ops = GenericOperations<Tensor, TensorBackend>;

public partial class Tensor
{
    public static Tensor Sum(Tensor a) => Ops.Sum(a);

    public static Tensor Sum(Tensor a, int[] axis) => Ops.Sum(a, axis);

    public static Tensor Outer(Tensor a, Tensor b)
    {
        if (!a.IsVector)
            throw new NotImplementedException("Only 1 dim outer product is supported");

        if (!b.IsVector)
            throw new NotImplementedException("Only 1 dim outer product is supported");

        return new(
            shape: [a.Size, b.Size],
            children: [a, b],
            forward: static self => TensorBackend.ExecuteOuter(self.A, self.B, self),
            backward: static (grad, self) => [
                Matmul(grad, self.A), 
                Matmul(grad.Transpose(), self.B)]);
    }

    public static Tensor Identity(Tensor a) => a;

    public static Tensor Sigmoid(Tensor a) =>
        new(shape: a.Shape,
            children: [a],
            forward: static self => TensorBackend.ExecuteSigmoid(self.A, self),
            backward: static (grad, self) => [grad * self * (1f - self)]);

    public static Tensor Tanh(Tensor a) =>
        new(shape: a.Shape,
            children: [a],
            forward: static self => TensorBackend.ExecuteTanh(self.A, self),
            backward: static (grad, self) => [grad * (1f - self * self)]);

    public static Tensor Matmul(Tensor a, Tensor b)
    {
        if (a.IsScalar || b.IsScalar)
            return Ops.Mul(a, b);

        if (a.IsVector && b.IsVector)
        {
            Shapes.EnsureShapesAreEqual(a.Shape, b.Shape);

            return new(
                [],
                children: [a, b],
                forward: static self => TensorBackend.ExecuteDot(self.A, self.B, self),
                backward: MatMulBackward);
        }
        
        if (a.IsVector)
        {
            if (a.Size != b.PrevDimension)
                throw new NotCompatibleShapesException(a.Shape, b.Shape);

            return new(
                b.Shape[1..],
                children: [a, b],
                forward: static self => TensorBackend.ExecuteVecMatMul(self.A, self.B.T, self),
                backward: MatMulBackward);
        }
        
        if (b.IsVector)
        {
            if (a.LastDimension != b.Size)
                throw new NotCompatibleShapesException(a.Shape, b.Shape);

            return new(
                a.Shape[..^1],
                children: [a, b],
                forward: static self => TensorBackend.ExecuteMatVecMul(self.A, self.B, self),
                backward: MatMulBackward);
        }

        
        if (a.LastDimension != b.PrevDimension)
            throw new NotCompatibleShapesException(a.Shape, b.Shape);

        if (a.IsRow && b.IsColumn)
        {
            return new(
                [1, 1],
                children: [a, b],
                forward: static self => TensorBackend.ExecuteDot(self.A, self.B.T, self),
                backward: MatMulBackward);
        }

        if (a.IsRow)
        {
            return new(
                [1, ..b.Shape[1..]],
                children: [a, b],
                forward: static self => TensorBackend.ExecuteVecMatMul(self.A, self.B.T, self),
                backward: MatMulBackward);
        }

        if (b.IsColumn)
        {
            return new(
                [..a.Shape[..^1], 1],
                children: [a, b],
                forward: static self => TensorBackend.ExecuteMatVecMul(self.A, self.B, self),
                backward: MatMulBackward);
        }

        var batchDimensions = Shapes.EnsureShapesAreCompatible(a.Shape[..^2], b.Shape[..^2]);

        return new(
            [..batchDimensions, a.PrevDimension, b.LastDimension],
            children: [a, b],
            forward: static self => TensorBackend.ExecuteMatMulTransposed(self.A, self.B.T, self),
            backward: MatMulBackward);
    }

    private static Tensor[] MatMulBackward(Tensor grad, Tensor self)
    {
        var aBatchShape = self.A.Dimensions <= 2 ? [] : self.A.Shape[..^2];
        var bBatchShape = self.B.Dimensions <= 2 ? [] : self.B.Shape[..^2];
        var rBatchShape = self.Dimensions <= 2 ? [] : self.Shape[..^2];

        var adims = Shapes.GetBroadcastedAxis(aBatchShape, rBatchShape);
        var bdims = Shapes.GetBroadcastedAxis(bBatchShape, rBatchShape);

        var agrad = grad.IsVector && self.B.IsVector
            ? Outer(grad, self.B.T)
            : Ops.Sum(Matmul(grad, self.B.T), axis: adims).Reshape(self.A.Shape);

        var bgrad = grad.IsVector && self.A.IsVector
            ? Outer(self.A.T, grad)
            : Ops.Sum(Matmul(self.A.T, grad), axis: bdims).Reshape(self.B.Shape);

        return
        [
            agrad,
            bgrad
        ];
    }
}
