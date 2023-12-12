namespace BitTensor.Core;

public partial class Tensor
{
    public static Tensor Add(Tensor a, Tensor b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Shape, b.Shape);

        return new(
            shape,
            children: [a, b],
            forward: static self => Ops.Add(self.A, self.B, self.Data),
            backward: static (grad, self) =>
            {
                var a_bc_dims = new List<int>(self.Dimensions);
                var b_bc_dims = new List<int>(self.Dimensions);

                for (var i = 1; i <= self.Dimensions; i++)
                {
                    if (i > self.A.Dimensions || self.A.Shape[^i] != self.Shape[^i])
                        a_bc_dims.Add(self.Dimensions - i);

                    if (i > self.B.Dimensions || self.B.Shape[^i] != self.Shape[^i]) 
                        b_bc_dims.Add(self.Dimensions - i);
                }

                var agrad = Sum(grad, axis: a_bc_dims.ToArray());
                var bgrad = Sum(grad, axis: b_bc_dims.ToArray());
                return
                [
                    agrad.Reshape(self.A.Shape),
                    bgrad.Reshape(self.B.Shape),
                ];
            });
    }

    public static Tensor Add(float a, Tensor b) => Add(b, a);

    public static Tensor Add(Tensor a, float b) =>
        new(shape: a.Shape,
            children: [a],
            forward: self => Ops.Add(self.A, b, self.Data),
            backward: static (grad, _) => [grad]);


    public static Tensor Negate(Tensor a) =>
        new(shape: a.Shape,
            children: [a],
            forward: static self => Ops.Negate(self.A, self.Data),
            backward: static (grad, _) => [-grad]);

    public static Tensor Mul(float a, Tensor b) => Mul(b, a);

    public static Tensor Mul(Tensor a, float b) =>
        b switch
        {
            0f => Zeros(a.Shape),
            1f => a,
            -1f => -a,
            _ => new(
                shape: a.Shape,
                children: [a],
                forward: self => Ops.Multiply(self.A, b, self.Data),
                backward: (grad, _) => [b * grad])
        };

    public static Tensor Mul(Tensor a, Tensor b)
    {
        if (ReferenceEquals(a, One))
            return b;

        if (ReferenceEquals(b, One))
            return a;

        if (ReferenceEquals(a, b))
            return Square(a);

        var shape = Shapes.EnsureShapesAreCompatible(a.Shape, b.Shape);
        
        return new(
            shape,
            children: [a, b],
            forward: static self => Ops.Multiply(self.A, self.B, self.Data),
            backward: static (grad, self) => [self.B * grad, self.A * grad]);
    }

    public static Tensor Square(Tensor a) =>
        new(shape: a.Shape,
            children: [a],
            forward: static self => Ops.Multiply(self.A, self.A, self.Data),
            backward: static (grad, self) => [grad * self.A * 2]);

    public static Tensor Outer(Tensor a, Tensor b)
    {
        if (!a.IsVector)
            throw new NotImplementedException("Only 1 dim outer product is supported");

        if (!b.IsVector)
            throw new NotImplementedException("Only 1 dim outer product is supported");

        return new(
            shape: [a.Size, b.Size],
            children: [a, b],
            forward: static self => Ops.Outer(self.A, self.B, self.Data),
            backward: static (grad, self) => [
                Matmul(grad, self.A), 
                Matmul(grad.Transpose(), self.B)]);
    }

    public static Tensor Pow(Tensor a, float power) =>
        power switch
        {
            0f => Ones(a.Shape),
            1f => a,
            2f => Square(a),
            _ => new(
                shape: a.Shape,
                children: [a],
                forward: self => Ops.Power(self.A, power, self.Data),
                backward: (grad, _) => [grad * power * Pow(a, power - 1)])
        };

    public static Tensor Sum(Tensor a) =>
        new(shape: [],
            children: [a],
            forward: static self => Ops.Sum(self.A, self.Data),
            backward: static (grad, self) => [Broadcast(grad, self.A.Shape)]);

    public static Tensor Sum(Tensor a, int[] axis) => Sum(a, new HashSet<int>(axis));

    private static Tensor Sum(Tensor a, HashSet<int> axis)
    {
        if (axis.Count == 0)
            return a;

        if (axis.Count == a.Dimensions)
            return Sum(a);

        return new(
            shape: a.Shape.Where((s, i) => !axis.Contains(i)).ToArray(),
            children: [a],
            forward: self => Ops.SumAxis(self.A, axis, self.Data),
            backward: Ops.NotSupported);
    }

    public static Tensor Broadcast(Tensor a, int[] shape)
    {
        if (!a.IsScalar)
            throw new NotImplementedException($"Not implemented for {a.Dimensions} dims");

        return new(
            shape: shape,
            children: [a],
            forward: static self => Ops.Broadcast(self.A, self.Data),
            backward: Ops.NotSupported);
    }

    public static Tensor Identity(Tensor a) => a;

    public static Tensor Sigmoid(Tensor a) =>
        new(shape: a.Shape,
            children: [a],
            forward: static self => Ops.Sigmoid(self.A, self.Data),
            backward: static (grad, self) => [grad * self * (1f - self)]);

    public static Tensor Tanh(Tensor a) =>
        new(shape: a.Shape,
            children: [a],
            forward: static self => Ops.Tanh(self.A, self.Data),
            backward: static (grad, self) => [grad * (1f - Square(self))]);

    public static Tensor Matmul(Tensor a, Tensor b)
    {
        if (a.IsScalar || b.IsScalar)
            return Mul(a, b);

        if (a.IsVector && b.IsVector)
        {
            Shapes.EnsureShapesAreEqual(a.Shape, b.Shape);

            return new(
                [],
                children: [a, b],
                forward: static self => Ops.Dot(self.A, self.B, self.Data),
                backward: MatMulBackward);
        }
        
        if (a.IsVector)
        {
            if (a.Size != b.NextToLastDimension)
                throw new NotCompatibleShapesException(a, b);

            return new(
                b.Shape[1..],
                children: [a, b],
                forward: static self => Ops.VecMatMul(self.A, self.B.T, self.Data),
                backward: MatMulBackward);
        }
        
        if (b.IsVector)
        {
            if (a.LastDimension != b.Size)
                throw new NotCompatibleShapesException(a, b);

            return new(
                a.Shape[..^1],
                children: [a, b],
                forward: static self => Ops.MatVecMul(self.A, self.B, self.Data),
                backward: MatMulBackward);
        }

        
        if (a.LastDimension != b.NextToLastDimension)
            throw new InvalidOperationException($"Shapes are incompatible: {a.Shape.Serialize()} and {b.Shape.Serialize()}");

        if (a.IsRow && b.IsColumn)
        {
            return new(
                [1, 1],
                children: [a, b],
                forward: static self => Ops.Dot(self.A, self.B.T, self.Data),
                backward: MatMulBackward);
        }

        if (a.IsRow)
        {
            return new(
                [1, ..b.Shape[1..]],
                children: [a, b],
                forward: static self => Ops.VecMatMul(self.A, self.B.T, self.Data),
                backward: MatMulBackward);
        }

        if (b.IsColumn)
        {
            return new(
                [..a.Shape[..^1], 1],
                children: [a, b],
                forward: static self => Ops.MatVecMul(self.A, self.B, self.Data),
                backward: MatMulBackward);
        }

        return new(
            [..a.Shape[..^1], b.LastDimension],
            children: [a, b],
            forward: static self => Ops.MatMulTransposed(self.A, self.B.T, self.Data),
            backward: MatMulBackward);
    }

    private static Tensor[] MatMulBackward(Tensor grad, Tensor self) =>
    [
        grad.IsVector && self.B.IsVector
            ? Outer(grad, self.B.T)
            : Matmul(grad, self.B.T),

        grad.IsVector && self.A.IsVector
            ? Outer(self.A.T, grad)
            : Matmul(self.A.T, grad)
    ];
}

public class NotCompatibleShapesException : Exception
{
    public NotCompatibleShapesException(Tensor a, Tensor b) : 
        base($"Shapes are incompatible: {a.Shape.Serialize()} and {b.Shape.Serialize()}") { }
}
