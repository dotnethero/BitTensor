using System.Numerics.Tensors;

namespace BitTensor.Core;

internal delegate void TensorTensorOperation(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> r);
internal delegate void TensorScalarOperation(ReadOnlySpan<float> a, float b, Span<float> r);

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
        BroadcastBinary(a, b, result, TensorPrimitives.Add, TensorPrimitives.Add);
    }

    public static void Multiply(Tensor a, float b, float[] result)
    {
        TensorPrimitives.Multiply(a.Values, b, result);
    }
    
    public static void Multiply(Tensor a, Tensor b, float[] result)
    {
        BroadcastBinary(a, b, result, TensorPrimitives.Multiply, TensorPrimitives.Multiply);
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

    public static void Sum(Tensor a, float[] result)
    {
        result[0] = TensorPrimitives.Sum(a.Values);
    }

    public static void SumAxis(Tensor a, HashSet<int> axis, float[] result)
    {
        var dims = a.Dimensions;
        var left = 0;
        var right = 0;

        for (var m = 0; m < dims && axis.Contains(m); m++) 
            ++left;

        for (var m = dims - 1; m >= 0 && axis.Contains(m); m--) 
            ++right;

        if (right == dims)
        {
            Sum(a, result);
            return;
        }

        a.EnsureHasUpdatedValues();
        
        if (right == axis.Count)
        {
            ReduceRight(a.Data, a.Shape, right, result);
            return;
        }
        
        if (left == axis.Count)
        {
            ReduceLeft(a.Data, a.Shape, left, result);
            return;
        }

        // TODO: decide on the fly what is faster - left or right first
        
        if (right + left == axis.Count)
        {
            var reduced = a.Shape[..^right];
            var next = new float[reduced.Product()];
            ReduceRight(a.Data, a.Shape, right, next);
            ReduceLeft(next, reduced, left, result);
            return;
        }

        var temp = a.Data;
        var shape = a.Shape;

        if (right != 0)
        {
            var reduced = shape[..^right];
            var next = new float[reduced.Product()];
            ReduceRight(temp, shape, right, next);
            temp = next;
            shape = reduced;
        }

        if (left != 0)
        {
            var reduced = shape[left..];
            var next = new float[reduced.Product()];
            ReduceLeft(temp, shape, left, next);
            temp = next;
            shape = reduced;
        }

        var axisAfterReduce = axis.Select(ax => ax - left).ToHashSet();
        SumNaive(temp, shape, axisAfterReduce, result);
    }

    private static void SumNaive(float[] data, int[] shape, HashSet<int> axis, float[] result)
    {
        Array.Clear(result);

        var size = data.Length;
        var dims = shape.Length;

        var old_strides = shape.GetStrides();
        var new_strides = new int[dims];
        var new_stride = 1;
        for (var m = dims - 1; m >= 0; --m)
        {
            if (!axis.Contains(m))
            {
                new_strides[m] = new_stride;
                new_stride *= shape[m];
            }
        }

        fixed (float* ap = data, rp = result)
        {
            for (var i = 0; i < size; i++)
            {
                var index = 0;
                var temp = i;
                for (var m = 0; m < dims; m++)
                {
                    var dim_old = temp / old_strides[m];
                    var dim_new = axis.Contains(m) ? 0 : dim_old;
                    temp -= dim_old * old_strides[m];
                    index += dim_new * new_strides[m];
                }

                rp[index] += ap[i];
            }
        }
    }

    private static void ReduceLeft(float[] data, int[] shape, int dimensions, float[] result)
    {
        Array.Clear(result);

        var count = shape[..dimensions].Product();
        var size = shape[dimensions..].Product();
        var values = data.AsSpan();

        for (var i = 0; i < count; i++)
        {
            var slice = values.Slice(i * size, size);
            TensorPrimitives.Add(result, slice, result);
        }
    }
    
    private static void ReduceRight(float[] data, int[] shape, int dimensions, float[] result)
    {
        var count = shape[..^dimensions].Product();
        var size = shape[^dimensions..].Product();
        var values = data.AsSpan();

        fixed (float* rp = result)
        {
            for (var i = 0; i < count; i++)
            {
                var slice = values.Slice(i * size, size);
                rp[i] = TensorPrimitives.Sum(slice);
            }
        }
    }

    public static void Broadcast(Tensor a, float[] result) // TODO: support axis
    {
        Array.Fill(result, a.Values[0]);
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
    
    public static void ApplyTransposeMatrix(float[] source, int[] matrix, float[] result)
    {
        var size = result.Length;

        fixed (float* r = result, s = source)
        fixed (int* m = matrix)
            for (var i = 0; i < size; i++)
            {
                r[i] = s[m[i]];
            }
    }
    
    private static void BroadcastBinary(Tensor a, Tensor b, float[] result, TensorTensorOperation tensorOp, TensorScalarOperation scalarOp)
    {
        var total = Math.Max(a.Dimensions, b.Dimensions);
        var ars = new int[total]; // reversed shapes
        var brs = new int[total];
        var rrs = new int[total];
        var dims = 0;

        for (var i = 0; i < total; ++i)
        {
            var ai = i >= a.Dimensions ? 1 : a.Shape[^(i+1)];
            var bi = i >= b.Dimensions ? 1 : b.Shape[^(i+1)];
            var ri = ai >= bi ? ai : bi;

            ars[dims] = ai;
            brs[dims] = bi;
            rrs[dims] = ri;

            ++dims;
        }
        
        var a_ones = 0;
        var b_ones = 0;
        var sames = 0;

        for (var i = 0; i < dims && ars[i] == 1; i++) 
            a_ones++;
        
        for (var i = 0; i < dims && brs[i] == 1; i++) 
            b_ones++;

        for (var i = 0; i < dims && ars[i] == brs[i]; i++) 
            sames++;

        var ones = Math.Max(a_ones, b_ones);
        var vdims = Math.Max(sames, ones); // dimensions to vectorize
        
        if (a_ones > b_ones)
        {
            (a, b) = (b, a);
            (ars, brs) = (brs, ars);
        }

        var vstride = rrs[..vdims].Product();

        var a_strides = new int[dims - vdims];
        var b_strides = new int[dims - vdims];
        var r_strides = new int[dims - vdims];

        if (dims > vdims) // else: full vector
        {
            a_strides[0] = 1;
            b_strides[0] = 1;
            r_strides[0] = 1;

            for (var i = 1; i < dims - vdims; ++i)
            {
                a_strides[i] = a_strides[i - 1] * ars[i + vdims - 1];
                b_strides[i] = b_strides[i - 1] * brs[i + vdims - 1];
                r_strides[i] = r_strides[i - 1] * rrs[i + vdims - 1];
            }

            for (var i = 0; i < dims - vdims; ++i)
            {
                if (ars[i + vdims] == 1)
                    a_strides[i] = 0;

                if (brs[i + vdims] == 1)
                    b_strides[i] = 0;
            }
        }

        var a_span = a.Values;
        var b_span = b.Values;
        var r_span = result.AsSpan();
        var r_count = rrs[vdims..].Product();

        for (var ri = 0; ri < r_count; ri++)
        {
            var ai = 0;
            var bi = 0;
            var leftover = ri;
            for (var i = dims - vdims - 1; i >= 0; --i)
            {
                var di = leftover / r_strides[i]; // dimension index
                ai += a_strides[i] * di;
                bi += b_strides[i] * di;
                leftover -= di * r_strides[i];
            }

            if (ones > sames) // vectorize scalar
            {
                var aslice = a_span.Slice(ai * vstride, vstride);;
                var bslice = b_span[bi];
                var rslice = r_span.Slice(ri * vstride, vstride);
                scalarOp(aslice, bslice, rslice);
            }
            else // vectorize same part
            {
                var aslice = a_span.Slice(ai * vstride, vstride);
                var bslice = b_span.Slice(bi * vstride, vstride);
                var rslice = r_span.Slice(ri * vstride, vstride);
                tensorOp(aslice, bslice, rslice);
            }
        }
    }

    internal static Tensor[] NotSupported(Tensor grad, Tensor self)
    {
        throw new NotSupportedException("Operations is not supported");
    }
}
