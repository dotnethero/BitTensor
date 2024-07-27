namespace BitTensor.Abstractions;

public static class GenericOperations<TTensor, TBackend>
    where TTensor : AbstractTensorNode<TTensor>, ITensorNode<TTensor>, ITensor<TTensor>
    where TBackend : ITensorBackend<TTensor>
{
    public static TTensor Negate(TTensor a) =>
        TTensor.CreateNode(
            shape: a.Shape,
            children: [a],
            forward: static self => TBackend.ExecuteNegate(self.A, self),
            backward: static (grad, _) => [Negate(grad)]);

    public static TTensor Add(TTensor a, TTensor b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Shape, b.Shape);

        return TTensor.CreateNode(
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

    public static TTensor Add(float a, TTensor b) => Add(b, a);

    public static TTensor Add(TTensor a, float b) =>
        TTensor.CreateNode(
            shape: a.Shape,
            children: [a],
            forward: self => TBackend.ExecuteAdd(self.A, b, self),
            backward: static (grad, _) => [grad]);

    public static TTensor Mul(float a, TTensor b) => Mul(b, a);

    public static TTensor Mul(TTensor a, float b) =>
        TTensor.CreateNode(
            shape: a.Shape,
            children: [a],
            forward: self => TBackend.ExecuteMultiply(self.A, b, self),
            backward: (grad, _) => [Mul(b, grad)]);

    public static TTensor Mul(TTensor a, TTensor b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Shape, b.Shape);

        return TTensor.CreateNode(
            shape,
            children: [a, b],
            forward: static self => TBackend.ExecuteMultiply(self.A, self.B, self),
            backward: static (grad, self) =>
            [
                Mul(self.B, grad),
                Mul(self.A, grad)
            ]);
    }

    public static TTensor Reshape(TTensor a, int[] shape)
    {
        if (shape.Product() != a.Size)
            throw new InvalidOperationException($"Shape {shape.Serialize()} does not produce {a.Size} size");

        return TTensor.CreateReshape(shape, a);
    }

    public static TTensor Broadcast(TTensor a, int[] shape)
    {
        if (!a.IsScalar)
            throw new NotImplementedException($"Not implemented for {a.Dimensions} dims");

        return TTensor.CreateNode(
            shape: shape,
            children: [a],
            forward: static self => TBackend.ExecuteBroadcast(self.A, self),
            backward: NotSupported);
    }

    public static TTensor Sum(TTensor a) =>
        TTensor.CreateNode(
            shape: [],
            children: [a],
            forward: static self => TBackend.ExecuteSum(self.A, self),
            backward: static (grad, self) => [Broadcast(grad, self.A.Shape)]);

    private static TTensor Sum(TTensor a, HashSet<int> axis)
    {
        if (axis.Count == 0)
            return a;

        if (axis.Count == a.Dimensions)
            return Sum(a);

        return TTensor.CreateNode(
            shape: a.Shape.Where((s, i) => !axis.Contains(i)).ToArray(),
            children: [a],
            forward: self => TBackend.ExecuteSum(self.A, axis, self),
            backward: NotSupported);
    }

    public static TTensor Sum(TTensor a, int[] axis) => Sum(a, new HashSet<int>(axis));

    public static TTensor[] NotSupported(TTensor grad, TTensor self) => throw new NotSupportedException("Operation is not supported");
}