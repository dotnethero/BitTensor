using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace BitTensor.Core;

internal delegate void TensorTensorOperation(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> r);
internal delegate void TensorScalarOperation(ReadOnlySpan<float> a, float b, Span<float> r);

internal static unsafe class Ops
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Negate(Tensor a, Tensor result)
    {
        TensorPrimitives.Negate(a.Values, result.Data);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Sigmoid(Tensor a, Tensor result)
    {
        TensorPrimitives.Sigmoid(a.Values, result.Data);
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Tanh(Tensor a, Tensor result)
    {
        TensorPrimitives.Tanh(a.Values, result.Data);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Add(Tensor a, float b, Tensor result)
    {
        TensorPrimitives.Add(a.Values, b, result.Data);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Add(Tensor a, Tensor b, Tensor result)
    {
        BroadcastBinary(a, b, result, TensorPrimitives.Add, TensorPrimitives.Add);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Multiply(Tensor a, float b, Tensor result)
    {
        TensorPrimitives.Multiply(a.Values, b, result.Data);
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Multiply(Tensor a, Tensor b, Tensor result)
    {
        BroadcastBinary(a, b, result, TensorPrimitives.Multiply, TensorPrimitives.Multiply);
    }

    public static void Power(Tensor a, float power, Tensor result)
    {
        fixed (float* ap = a.Values, rp = result.Data)
        {
            for (var i = 0; i < a.Size; i++)
            {
                rp[i] = MathF.Pow(ap[i], power);
            }
        }
    }
    
    public static void Outer(Tensor a, Tensor b, Tensor result)
    {
        var span = result.Data.AsSpan();

        fixed (float* ap = a.Values)
        {
            for (var i = 0; i < a.Size; i++)
            {
                TensorPrimitives.Multiply(b.Values, ap[i], span[(i * b.Size)..]);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Sum(Tensor a, Tensor result)
    {
        result.Data[0] = TensorPrimitives.Sum(a.Values);
    }

    public static void SumAxis(Tensor a, HashSet<int> axis, Tensor result)
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
            ReduceRight(a.Data, a.Shape, right, result.Data);
            return;
        }
        
        if (left == axis.Count)
        {
            ReduceLeft(a.Data, a.Shape, left, result.Data);
            return;
        }

        // TODO: decide on the fly what is faster - left or right first
        
        if (right + left == axis.Count)
        {
            var reduced = a.Shape[..^right];
            var next = new float[reduced.Product()];
            ReduceRight(a.Data, a.Shape, right, next);
            ReduceLeft(next, reduced, left, result.Data);
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
        SumNaive(temp, shape, axisAfterReduce, result.Data);
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Broadcast(Tensor a, Tensor result) // TODO: support axis
    {
        Array.Fill(result.Data, a.Values[0]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Dot(Tensor a, Tensor b, Tensor result)
    {
        result.Data[0] = TensorPrimitives.Dot(a.Values, b.Values);
    }

    public static void MatVecMul(Tensor a, Tensor b, Tensor result)
    {
        var (batchCount, rowCount, rowSize) = Shapes.GetBatchRowsColumns(a.Shape);

        a.EnsureHasUpdatedValues();
        b.EnsureHasUpdatedValues();

        var col = b.Data.AsSpan();

        fixed (float* rp = result.Data)
        {
            for (var batchIndex = 0; batchIndex < batchCount; batchIndex++)
            {
                var batchSize = rowSize * rowCount;
                var batchStride = batchIndex * rowCount;
                var batch = a.Data.AsSpan(batchIndex * batchSize, batchSize);

                for (var rowIndex = 0; rowIndex < rowCount; rowIndex++)
                {
                    var row = batch.Slice(rowIndex * rowSize, rowSize);
                    var dot = TensorPrimitives.Dot(row, col);
                    rp[batchStride + rowIndex] = dot;
                }
            }
        }
    }

    public static void VecMatMul(Tensor a, Tensor bT, Tensor result)
    {
        var (batchCount, colCount, rowSize) = Shapes.GetBatchRowsColumns(bT.Shape);

        a.EnsureHasUpdatedValues();
        bT.EnsureHasUpdatedValues();

        var row = a.Data.AsSpan();

        fixed (float* rp = result.Data)
        {
            for (var batchIndex = 0; batchIndex < batchCount; batchIndex++)
            {
                var batchSize = colCount * rowSize;
                var batchStride = batchIndex * colCount;
                var batch = bT.Data.AsSpan(batchIndex * batchSize, batchSize);

                for (var colIndex = 0; colIndex < colCount; ++colIndex)
                {
                    var col = batch.Slice(colIndex * rowSize, rowSize);
                    var dot = TensorPrimitives.Dot(row, col);
                    rp[batchStride + colIndex] = dot;
                }
            }
        }
    }

    public static void MatMulTransposed(Tensor a, Tensor bT, Tensor result)
    {
        a.EnsureHasUpdatedValues();
        bT.EnsureHasUpdatedValues();

        var strides = Batching.GetBatchStrides(a, bT);

        ParallelOptions options = new();
        Parallel.ForEach(GetMatMulAtoms(strides, a, bT, result), options, MatMulRow);
    }

    private readonly record struct MatMulAtom(
        Tensor A,
        Tensor B,
        Tensor Result,
        int BatchIndexA = 0,
        int BatchIndexB = 0,
        int BatchIndexR = 0,
        int RowIndex = 0);

    private static IEnumerable<MatMulAtom> GetMatMulAtoms(BatchStrides strides, Tensor a, Tensor b, Tensor r)
    {
        var rowCount = a.Shape[^2];
        var iterator = new MatMulAtom(a, b, r);
        for (var batchIndex = 0; batchIndex < strides.BatchCount; batchIndex++)
        {
            var (aIndex, bIndex) = strides.ConvertIndex(batchIndex);
            for (var rowIndex = 0; rowIndex < rowCount; rowIndex++)
            {
                yield return iterator with
                {
                    BatchIndexA = aIndex,
                    BatchIndexB = bIndex,
                    BatchIndexR = batchIndex,
                    RowIndex = rowIndex
                };
            }
        }
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MatMulRow(MatMulAtom inputs)
    {
        var rowSize = inputs.A.Shape[^1];
        var rowCount = inputs.A.Shape[^2];
        var colCount = inputs.B.Shape[^2];

        var leftSize = rowCount * rowSize;
        var rightSize = colCount * rowSize;
        var batchSize = rowCount * colCount;

        var left = inputs.A.Data.AsSpan(inputs.BatchIndexA * leftSize, leftSize);
        var right = inputs.B.Data.AsSpan(inputs.BatchIndexB * rightSize, rightSize);

        fixed (float* rp = inputs.Result.Data)
        {
            var row = left.Slice(inputs.RowIndex * rowSize, rowSize);

            for (var colIndex = 0; colIndex < colCount; ++colIndex)
            {
                var col = right.Slice(colIndex * rowSize, rowSize);
                var dot = TensorPrimitives.Dot(row, col);
                rp[inputs.BatchIndexR * batchSize + inputs.RowIndex * colCount + colIndex] = dot;
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
    
    public static void ApplyTransposeMatrix(ReadOnlySpan<float> source, int[] matrix, Span<float> result)
    {
        var size = result.Length;

        fixed (float* r = result, s = source)
        fixed (int* m = matrix)
            for (var i = 0; i < size; i++)
            {
                r[i] = s[m[i]];
            }
    }
    
    private static void BroadcastBinary(Tensor a, Tensor b, Tensor result, TensorTensorOperation tensorOp, TensorScalarOperation scalarOp)
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
        var r_span = result.Data.AsSpan();
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
                var aslice = a_span.Slice(ai * vstride, vstride);
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