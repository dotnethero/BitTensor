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
                    agrad,
                    bgrad,
                ];
            });
    }

    public static Tensor ReduceLeft(Tensor a, int dimensions) =>
        (a.Dimensions - dimensions) switch
        {
            <0 => throw new InvalidOperationException($"Too much dimensions to reduce: {dimensions}"),
            0 => Sum(a),
            _ => new(
                shape: a.Shape[dimensions..],
                children: [a],
                forward: self => Ops.ReduceLeft(self.A, dimensions, self.Data),
                backward: static (grad, self) => [grad]) // Not sure about this
        };

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
        if (a.Dimensions != 1)
            throw new NotImplementedException("Only 1 dim outer product is supported");

        if (b.Dimensions != 1)
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

    public static Tensor Sum(Tensor a) => // TODO: axis
        new(shape: [],
            children: [a],
            forward: static self => Ops.Sum(self.A, self.Data),
            backward: static (grad, self) => [Broadcast(grad, self.A.Shape)]);
    
    public static Tensor Sum(Tensor a, int[] axis) =>
        new(shape: a.Shape.Where((s, i) => !axis.Contains(i)).ToArray(),
            children: [a],
            forward: self => Ops.SumAxis(self.A, axis, self.Data),
            backward: static (grad, self) => [Broadcast(grad, self.A.Shape)]); // TODO: double check
    
    public static Tensor SumKeepDims(Tensor a, int[] axis) =>
        new(shape: a.Shape.Select((s, i) => axis.Contains(i) ? 1 : s).ToArray(),
            children: [a],
            forward: self => Ops.SumAxis(self.A, axis, self.Data),
            backward: static (grad, self) => [Broadcast(grad, self.A.Shape)]); // TODO: double check

    public static Tensor Broadcast(Tensor a, int[] shape)
    {
        if (a.Dimensions > 0)
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
        // a shape: b1 * b2 * n * m
        // b shape: b1 * b2 * m * k

        if (a.Dimensions == 0 || b.Dimensions == 0)
        {
            return Mul(a, b);
        }

        var shrinkStart = 0;
        var shrinkEnd = 0;

        var ao = a;
        var bo = b;

        if (a.Dimensions == 1)
        {
            a = a.PrependDimension();
            shrinkStart = 1;
        }

        if (b.Dimensions == 1)
        {
            b = b.AppendDimension();
            shrinkEnd = 1;
        }

        if (a.Shape[^1] != b.Shape[^2])
            throw new InvalidOperationException($"MATMUL: Shapes are incompatible: {a.Shape.Serialize()} and {b.Shape.Serialize()}");

        var bT = b.Transpose();

        int[] shape = [..a.Shape[..^1], b.Shape[^1]];

        shape = shape[shrinkStart..^shrinkEnd];

        return new(
            shape,
            children: [a, b],
            forward: self => Ops.MatMulTransposed(a, bT, self.Data), // closure b transposed
            backward: MatMulBackward);

        Tensor[] MatMulBackward(Tensor grad, Tensor local)
        {
            var da =
                bo.Dimensions == 1 &&
                grad.Dimensions == 1
                    ? Outer(grad, bo.Transpose())
                    : Matmul(grad, bo.Transpose());

            var db =
                ao.Dimensions == 1 &&
                grad.Dimensions == 1
                    ? Outer(ao.Transpose(), grad)
                    : Matmul(ao.Transpose(), grad);

            return [da, db];
        }
    }
}
