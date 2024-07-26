namespace BitTensor.Abstractions;

public static class GenericOperations<T, TBackend>
    where T : AbstractTensorNode<T>, ITensorNode<T>, ITensor<T>
    where TBackend : ITensorBackend<T>
{
    public static T Identity(T a) => a;
    
    public static T Negate(T a) =>
        T.Create(
            shape: a.Shape,
            children: [a],
            forward: static self => TBackend.ExecuteNegate(self.A, self),
            backward: static (grad, _) => [Negate(grad)]);

    public static T Add(T a, T b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Shape, b.Shape);

        return T.Create(
            shape,
            children: [a, b],
            forward: static self => TBackend.ExecuteAdd(self.A, self.B, self),
            backward: static (grad, self) =>
            {
                var adims = Shapes.GetBroadcastedAxis(self.A.Shape, self.Shape);
                var bdims = Shapes.GetBroadcastedAxis(self.B.Shape, self.Shape);
                var agrad = Sum(grad, axis: adims);
                var bgrad = Sum(grad, axis: bdims);
                return
                [
                    Reshape(agrad, self.A.Shape),
                    Reshape(bgrad, self.B.Shape),
                ];
            });
    }

    public static T Add(float a, T b) => Add(b, a);

    public static T Add(T a, float b) =>
        T.Create(
            shape: a.Shape,
            children: [a],
            forward: self => TBackend.ExecuteAdd(self.A, b, self),
            backward: static (grad, _) => [grad]);

    public static T Mul(float a, T b) => Mul(b, a);

    public static T Mul(T a, float b) =>
        b switch
        {
            0f => T.Zeros(a.Shape),
            1f => a,
            _ => T.Create(
                shape: a.Shape,
                children: [a],
                forward: self => TBackend.ExecuteMultiply(self.A, b, self),
                backward: (grad, _) => [Mul(b, grad)])
        };

    public static T Mul(T a, T b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Shape, b.Shape);

        return T.Create(
            shape,
            children: [a, b],
            forward: static self => TBackend.ExecuteMultiply(self.A, self.B, self),
            backward: static (grad, self) =>
            [
                Mul(self.B, grad),
                Mul(self.A, grad)
            ]);
    }

    public static T Pow(T a, float power) =>
        power switch
        {
            0f => T.Ones(a.Shape),
            1f => Identity(a),
            _ => T.Create(
                shape: a.Shape,
                children: [a],
                forward: self => TBackend.ExecutePower(self.A, power, self),
                backward: (grad, _) => [PowBackward(grad, a, power)])
        };

    private static T PowBackward(T grad, T a, float power) =>
        Mul(power, Mul(grad, Pow(a, power - 1)));

    public static T Reshape(T a, int[] shape)
    {
        if (shape.Product() != a.Size)
            throw new InvalidOperationException($"Shape {shape.Serialize()} does not produce {a.Size} size");

        return T.Create(
            shape,
            children: [a],
            forward: static self => TBackend.ExecuteReshape(self.A, self),
            backward: (grad, _) => [Reshape(grad, a.Shape)]);
    }

    public static T Broadcast(T a, int[] shape)
    {
        if (!a.IsScalar)
            throw new NotImplementedException($"Not implemented for {a.Dimensions} dims");

        return T.Create(
            shape: shape,
            children: [a],
            forward: static self => TBackend.ExecuteBroadcast(self.A, self),
            backward: NotSupported);
    }

    public static T Sum(T a) =>
        T.Create(
            shape: [],
            children: [a],
            forward: static self => TBackend.ExecuteSum(self.A, self),
            backward: static (grad, self) => [Broadcast(grad, self.A.Shape)]);

    private static T Sum(T a, HashSet<int> axis)
    {
        if (axis.Count == 0)
            return a;

        if (axis.Count == a.Dimensions)
            return Sum(a);

        return T.Create(
            shape: a.Shape.Where((s, i) => !axis.Contains(i)).ToArray(),
            children: [a],
            forward: self => TBackend.ExecuteSum(self.A, axis, self),
            backward: NotSupported);
    }

    public static T Sum(T a, int[] axis) => Sum(a, new HashSet<int>(axis));

    public static T[] NotSupported(T grad, T self) => throw new NotSupportedException("Operation is not supported");
}