﻿namespace BitTensor.Core;

public partial class Tensor
{
    public Tensor this[params int[] indexes] => GetSlice(indexes);
    
    public Tensor AppendDimension() => Reshape([..Shape, 1]);

    public Tensor PrependDimension() => Reshape([1, ..Shape]);

    public Tensor Shuffle(int dimension, int[] permutation)
    {
        if (permutation.Length != Shape[dimension])
            throw new InvalidOperationException($"Invalid permutation size: {permutation.Length}, expected: {Shape[dimension]}");

        EnsureHasUpdatedValues();

        var data = new float[Size];
        var span = Data.AsSpan();
        var count = Shape.Take(dimension).Product();
        var dimSize = Shape.Skip(dimension + 1).Product();
        var dimCount = Shape[dimension];

        for (var i = 0; i < count; i++)
        for (var j = 0; j < dimCount; j++)
        {
            var k = permutation[j]; // new index
            var src = span.Slice(i * dimSize * dimCount + j * dimSize, dimSize);
            var dest = data.AsSpan(i * dimSize * dimCount + k * dimSize, dimSize);
            src.CopyTo(dest);
        }

        Array.Copy(data, Data, Size);
        Invalidate();
        return this;
    }

    public Tensor Reshape(int[] shape)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(shape.Product(), Size);

        return new(
            shape,
            children: [this],
            forward: t => {},
            backward: (grad, _) => [grad.Reshape(Shape)],
            values: Data);
    }
    
    public Tensor GetSlice(int[] indexes)
    {
        var range = GetSliceRange(indexes);

        return new(
            shape: Shape[indexes.Length..],
            children: [this],
            forward: t => {},
            backward: (grad, _) => [grad.GetSlice(Shape)], 
            values: Data[range]);
    }

    private Range GetSliceRange(int[] indexes)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(indexes.Length, Dimensions);

        var flat = 0;
        var shift = 1;
        var slice = 1;
        for (var i = Dimensions - 1; i >= 0; i--)
        {
            flat += i < indexes.Length ? indexes[i] * shift : 0;
            slice *= i >= indexes.Length ? Shape[i] : 1;
            shift *= Shape[i];
        }
        return new(flat, flat + slice);
    }

    public Tensor Transpose()
    {
        if (TransposeHint is not null)
            return TransposeHint;

        if (Dimensions < 2)
            return this;

        var axes = new int[Dimensions];
        for (var i = 0; i < Dimensions; i++)
        {
            axes[i] = i;
        }

        axes[^1] = Dimensions - 2;
        axes[^2] = Dimensions - 1;
        return Transpose(axes);
    }

    public Tensor Transpose(int[] axes)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(axes.Length, Dimensions);
        ArgumentOutOfRangeException.ThrowIfNotEqual(new HashSet<int>(axes).Count, axes.Length);
        
        if (Dimensions < 2)
            return this;

        var shape = new int[Dimensions];
        for (var i = 0; i < Dimensions; i++)
        {
            shape[i] = Shape[axes[i]];
        }
        
        if (shape.Count(ax => ax > 1) <= 1)
        {
            return Reshape(shape);
        }

        var matrix = Ops.GetTransposeMatrix(this, axes);

        return new(
            shape,
            children: [this],
            forward: self => Ops.ApplyTransposeMatrix(this.Data, matrix, self.Data),
            backward: (grad, self) => [grad.Transpose(axes)])
            {
                TransposeHint = this
            };
    }
}