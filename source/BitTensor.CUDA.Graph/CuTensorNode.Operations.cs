﻿using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

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
            backward: (grad, _) =>
            {
                var adims = Shapes.GetBroadcastedAxis(a.Shape, grad.Shape);
                var bdims = Shapes.GetBroadcastedAxis(b.Shape, grad.Shape);
                return
                [
                    Sum(grad, axis: adims, scale: beta).Reshape(a.Shape),
                    Sum(grad, axis: bdims, scale: 1f).Reshape(b.Shape)
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
            backward: (grad, _) =>
            {
                var agrad = ElementwiseProduct(grad, b);
                var bgrad = ElementwiseProduct(grad, a);
                var adims = Shapes.GetBroadcastedAxis(a.Shape, agrad.Shape);
                var bdims = Shapes.GetBroadcastedAxis(b.Shape, bgrad.Shape);
                return
                [
                    Sum(agrad, axis: adims).Reshape(a.Shape),
                    Sum(bgrad, axis: bdims).Reshape(b.Shape)
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
            backward: (grad, _) => [grad * b, a * grad]); // TODO: scale!
    }

    public static CuTensorNode MatrixProduct(CuTensorNode a, CuTensorNode b)
    {
        var shape = Shapes.BroadcastMatrixProduct(a.Shape, b.Shape); // desired shape
        var output = new CuTensor(shape); // true output

        var modA = PadLeft(a);
        var modB = PadRight(b);
        var modShape = Shapes.BroadcastMatrixProduct(modA.Shape, modB.Shape); // padded shape
        var modOutput = output.Reshape(modShape); // padded output

        var context = new CuTensorContext();
        var plan = new CuTensorMatrixProductPlan(context, modA.Tensor, modB.Tensor, modOutput);

        return new(
            output,
            children: [a, b],
            forward: () => plan.Execute(modA.Tensor, modB.Tensor, modOutput),
            backward: (grad, _) =>
            {
                var gpad = grad.Reshape(modShape);
                var da = gpad * Transpose(modB);
                var db = Transpose(modA) * gpad;
                
                var adims = Shapes.GetBroadcastedAxis(modA.Shape, da.Shape);
                var bdims = Shapes.GetBroadcastedAxis(modB.Shape, db.Shape);
                var agrad = Sum(da, axis: adims).Reshape(a.Shape);
                var bgrad = Sum(db, axis: bdims).Reshape(b.Shape);
                return [agrad, bgrad];
            });
    }

    public static CuTensorNode Sum(CuTensorNode a)
    {
        var output = new CuTensor([]);
        return new(
            output,
            children: [a],
            forward: () => CuBackend.Sum(a.Tensor, output),
            backward: (grad, _) => [Broadcast(grad, a.Shape)]);
    }
    
    public static CuTensorNode Sum(CuTensorNode a, HashSet<int> axis, float scale = 1f)
    {
        var shape = a.Shape.Reduce(axis);
        var output = new CuTensor(shape);
        return new(
            output,
            children: [a],
            forward: () => CuBackend.Sum(a.Tensor, axis, output, scale),
            backward: (grad, _) => [Broadcast(grad, a.Shape)]); // TODO: Verify!
    }
    
    public static CuTensorNode Broadcast(CuTensorNode a, Shape shape)
    {
        if (!a.Shape.CanBroadcastTo(shape))
            throw new InvalidOperationException($"Can't broadcast {a.Shape} to {shape}");

        var output = new CuTensor(shape);
        var axis = Shapes.GetBroadcastedAxis(a.Shape, shape);
        return new(
            output,
            children: [a],
            forward: () => CuBackend.Broadcast(a.Tensor, output),
            backward: (grad, _) => [Sum(grad, axis)]); // TODO: Verify!
    }

    public static CuTensorNode Sigmoid(CuTensorNode a)
    {
        var one = new CuTensor([], [1]).ToNode();
        var output = new CuTensor(a.Shape);
        return new(
            output,
            children: [a],
            forward: () => CuBackend.Sigmoid(a.Tensor, output),
            backward: (grad, self) => [grad * self * (one - self)]);
    }

    public static CuTensorNode Transpose(CuTensorNode a)
    {
        var axis = a.Shape.GetTransposeAxis();
        return Transpose(a, axis);
    }
    
    public static CuTensorNode Transpose(CuTensorNode a, int[] axis)
    {
        if (axis.Length != a.Tensor.Dimensions)
            throw new InvalidOperationException($"Axis {axis.ToText()} is not valid argument for {a.Shape} shape tensor");

        if (!axis.AllElementsAreUnique())
            throw new InvalidOperationException($"Axis {axis.ToText()} does not contain all axes for {a.Shape} shape tensor");

        var shape = a.Shape.Transpose(axis);
        var output = new CuTensor(shape);
        return new(
            output,
            children: [a],
            forward: () => CuBackend.Transpose(a.Tensor, axis, output),
            backward: (grad, _) => [Transpose(grad, axis)]); // TODO: Verify!
    }

    public CuTensorNode Reshape(Shape shape)
    {
        if (shape.ArraySize != Tensor.Size)
            throw new InvalidOperationException($"Can't reshape {Shape} into {shape}");

        return new(
            Tensor.Reshape(shape), // no allocation
            children: [this],
            forward: () => {},
            backward: (grad, _) => [grad.Reshape(Tensor.Shape)]);
    }
    
    public CuTensorNode PadLeft() => Reshape([1, ..Shape]);

    public CuTensorNode PadRight() => Reshape([..Shape, 1]);

    private static CuTensorNode PadLeft(CuTensorNode node) =>
        node.Tensor.IsVector
            ? node.PadLeft()
            : node;
    
    private static CuTensorNode PadRight(CuTensorNode node) =>
        node.Tensor.IsVector
            ? node.PadRight()
            : node;
}