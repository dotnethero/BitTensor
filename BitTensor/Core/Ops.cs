using System.Numerics.Tensors;

namespace BitTensor.Core;

internal static unsafe class Ops
{
    public static void Negate(Tensor a, float[] result)
    {
        TensorPrimitives.Negate(a.Values, result);
    }

    public static void Sigmoid(Tensor a, float[] result)
    {
        TensorPrimitives.Sigmoid(a.Values, result);
    }
    
    public static void Tanh(Tensor a, float[] result)
    {
        TensorPrimitives.Tanh(a.Values, result);
    }

    public static void Add(Tensor a, float b, float[] result)
    {
        TensorPrimitives.Add(a.Values, b, result);
    }

    public static void Add(Tensor a, Tensor b, float[] result)
    {
        if (a.Dimensions > b.Dimensions)
        {
            var stride = a.Shape[^b.Dimensions..].Product();
            var strides = a.Shape[..^b.Dimensions].Product();
            var aspan = a.Values;
            var rspan = result.AsSpan();
            for (var i = 0; i < strides; i++)
            {
                var aslice = aspan.Slice(i * stride, stride);
                var rslice = rspan.Slice(i * stride, stride);
                TensorPrimitives.Add(aslice, b.Values, rslice);
            }
        }
        
        if (a.Dimensions < b.Dimensions)
        {
            var stride = b.Shape[^a.Dimensions..].Product();
            var strides = b.Shape[..^a.Dimensions].Product();
            var span = b.Values;
            var rspan = result.AsSpan();
            for (var i = 0; i < strides; i++)
            {
                var bslice = span.Slice(i * stride, stride);
                var rslice = rspan.Slice(i * stride, stride);
                TensorPrimitives.Add(a.Values, bslice, rslice);
            }
        }

        if (a.Dimensions == b.Dimensions)
        {
            TensorPrimitives.Add(a.Values, b.Values, result);
        }
    }
    
    public static void Multiply(Tensor a, float b, float[] result)
    {
        TensorPrimitives.Multiply(a.Values, b, result);
    }

    public static void Multiply(Tensor a, Tensor b, float[] result)
    {
        switch (a.Size, b.Size)
        {
            case (1, 1):
                result[0] = a.Values[0] * b.Values[0];
                break;

            case (>1, 1):
                TensorPrimitives.Multiply(a.Values, b.Values[0], result);
                break;

            case (1, >1):
                TensorPrimitives.Multiply(b.Values, a.Values[0], result);
                break;

            default:
                TensorPrimitives.Multiply(a.Values, b.Values, result);
                break;
        }
    }

    public static void Power(Tensor a, float power, float[] result)
    {
        fixed (float* ap = a.Values, rp = result)
        {
            for (var i = 0; i < a.Size; i++)
            {
                rp[i] = MathF.Pow(ap[i], power);
            }
        }
    }
    
    public static void Outer(Tensor a, Tensor b, float[] result)
    {
        var span = result.AsSpan();

        fixed (float* ap = a.Values)
        {
            for (var i = 0; i < a.Size; i++)
            {
                TensorPrimitives.Multiply(b.Values, ap[i], span[(i * b.Size)..]);
            }
        }
    }

    public static void Sum(Tensor a, float[] result) // TODO: support axis
    {
        result[0] = TensorPrimitives.Sum(a.Values);
    }
    
    public static void Broadcast(Tensor a, float[] result) // TODO: support axis
    {
        Array.Fill(result, a.Values[0]);
    }

    public static void ReduceLeft(Tensor a, int dimensions, float[] result)
    {
        Array.Clear(result);
        var count = a.Shape[..dimensions].Product();
        var size = a.Shape[dimensions..].Product();
        for (var i = 0; i < count; i++)
        {
            var slice = a.Values.Slice(i * size, size);
            TensorPrimitives.Add(result, slice, result);
        }
    }

    public static void MatMulTransposed(Tensor a, Tensor bT, float[] results)
    {
        var rowCount = a.Shape[^2];
        var batchDim = a.Shape[..^2];
        var batchCount = batchDim.Product();

        a.EnsureHasUpdatedValues();
        bT.EnsureHasUpdatedValues();

        if (batchCount * rowCount < 16)
        {
            for (var batchIndex = 0; batchIndex < batchCount; batchIndex++)
            {
                for (var rowIndex = 0; rowIndex < rowCount; rowIndex++)
                {
                    MatMulRow(a, bT, batchIndex, rowIndex, results);
                }
            }
        }
        else
        {
            Parallel.For(0, batchCount * rowCount, i => MatMulRowP(a, bT, i, rowCount, results));
        }
    }

    private static void MatMulRowP(Tensor a, Tensor bT, int compoundIndex, int rowCount, float[] results)
    {
        var batchIndex = compoundIndex / rowCount;
        var rowIndex = compoundIndex - rowCount * batchIndex;
        MatMulRow(a, bT, batchIndex, rowIndex, results);
    }

    private static void MatMulRow(Tensor a, Tensor bT, int batchIndex, int rowIndex, float[] results)
    {
        var rowSize = a.Shape[^1];
        var rowCount = a.Shape[^2];
        var colCount = bT.Shape[^2];

        var leftSize = rowCount * rowSize;
        var rightSize = colCount * rowSize;
        var batchSize = rowCount * colCount;

        var left = a.Data.AsSpan(batchIndex * leftSize, leftSize);
        var right = bT.Data.AsSpan(batchIndex * rightSize, rightSize);

        fixed (float* rp = results)
        {
            var row = left.Slice(rowIndex * rowSize, rowSize);

            for (var colIndex = 0; colIndex < colCount; ++colIndex)
            {
                var col = right.Slice(colIndex * rowSize, rowSize);
                var dot = TensorPrimitives.Dot(row, col);
                rp[batchIndex * batchSize + rowIndex * colCount + colIndex] = dot;
            }
        }
    }
    
    public static int[] GetTransposeMatrix(Tensor tensor, int[] axes)
    {
        var dims = tensor.Dimensions;
        var size = tensor.Size;

        // Compute the strides for the original and transposed tensor
        var originStrides = stackalloc int[dims];
        var resultStrides = stackalloc int[dims];
        originStrides[dims - 1] = 1;
        resultStrides[axes[dims - 1]] = 1;
        for (var i = dims - 2; i >= 0; i--)
        {
            originStrides[i] = originStrides[i + 1] * tensor.Shape[i + 1];
            resultStrides[axes[i]] = resultStrides[axes[i + 1]] * tensor.Shape[axes[i + 1]];
        }

        var matrix = new int[size];

        fixed (int* m = matrix)
            for (var i = 0; i < size; i++)
            {
                var resultIndex = 0;
                var temp = i;
                for (var j = 0; j < dims; j++)
                {
                    var quotient = temp / originStrides[j];
                    temp -= quotient * originStrides[j];
                    resultIndex += quotient * resultStrides[j];
                }

                m[resultIndex] = i;
            }

        return matrix;
    }
    
    public static void ApplyMatrix(float[] source, int[] matrix, float[] result)
    {
        var size = result.Length;

        fixed (float* r = result, s = source)
        fixed (int* m = matrix)
            for (var i = 0; i < size; i++)
            {
                r[i] = s[m[i]];
            }
    }

    internal static Tensor[] NotSupported(Tensor grad, Tensor self) => 
        throw new NotSupportedException("Operations is not supported");
}
