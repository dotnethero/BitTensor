namespace BitTensor.Core;

public partial class Tensor
{
    public Tensor this[params int[] indexes] => GetSlice(indexes);
    
    public Tensor AppendDimension() => Reshape([..Shape, 1]);

    public Tensor PrependDimension() => Reshape([1, ..Shape]);

    public unsafe Tensor Shuffle(int dimension, int[] permutation)
    {
        if (permutation.Length != Shape[dimension])
            throw new InvalidOperationException($"Invalid permutation size: {permutation.Length}, expected: {Shape[dimension]}");

        EnsureHasUpdatedValues();

        var data = new float[Size];
        var span = Data.AsSpan();
        var count = Shape[..dimension].Product();
        var dimSize = Shape[(dimension+1)..].Product();
        var dimCount = Shape[dimension];

        fixed(int* p = permutation)
            for (var i = 0; i < count; i++)
            for (var j = 0; j < dimCount; j++)
            {
                var k = p[j]; // new index
                var src = span.Slice(i * dimSize * dimCount + j * dimSize, dimSize);
                var dest = data.AsSpan(i * dimSize * dimCount + k * dimSize, dimSize);
                src.CopyTo(dest);
            }

        Data = data;
        Invalidate();
        return this;
    }

    public Tensor Reshape(int[] shape)
    {
        if (shape.Product() != Size)
            throw new InvalidOperationException($"Shape {shape.Serialize()} does not produce {Size} size");

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
        if (indexes.Length > Dimensions)
            throw new InvalidOperationException($"Index {indexes.Serialize()} is not valid for {Shape.Serialize()} shape");

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

        var dims = Dimensions;
        if (dims < 2)
            return this;

        var axes = new int[dims];
        for (var i = 0; i < dims; i++)
        {
            axes[i] = i;
        }

        axes[^1] = dims - 2;
        axes[^2] = dims - 1;
        return Transpose(axes);
    }

    public unsafe Tensor Transpose(int[] axes)
    {
        var dims = Dimensions;
        if (dims < 2)
            return this;

        if (axes.Length != dims)
            throw new InvalidOperationException($"Axis {axes.Serialize()} is not valid argument for {Shape.Serialize()} shape tensor");

        if (!axes.IsElementsUnique())
            throw new InvalidOperationException($"Axis {axes.Serialize()} does not contain all axes for {Shape.Serialize()} shape tensor");

        var shape = new int[dims];
        var countNon1 = 0;
        fixed (int* sh = Shape, sh_new = shape, ax = axes)
        {
            for (var i = 0; i < dims; i++)
            {
                var d = sh[ax[i]];
                if (d > 1) 
                    ++countNon1;

                sh_new[i] = d;
            }
        }

        if (countNon1 <= 1)
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