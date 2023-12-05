namespace BitTensor.Core;

public partial class Tensor
{
    public Tensor this[params int[] indexes] => GetSlice(indexes);

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
            backward: Ops.NotSupported, 
            values: Data[range]);
    }

    public Tensor AppendDimension() => Reshape([..Shape, 1]);

    public Tensor PrependDimension() => Reshape([1, ..Shape]);

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
        if (TransposeHint is not null)
            return TransposeHint;

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
            forward: t => Ops.ApplyMatrix(this.Data, matrix, t.Data),
            backward: Ops.NotSupported)
            {
                TransposeHint = this
            };
    }
}