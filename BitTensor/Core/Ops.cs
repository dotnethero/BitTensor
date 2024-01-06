using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using BitTensor.Playground;

namespace BitTensor.Core;

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
        Broadcasting.Binary<AddOperator>(a, b, result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Multiply(Tensor a, float b, Tensor result)
    {
        TensorPrimitives.Multiply(a.Values, b, result.Data);
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Multiply(Tensor a, Tensor b, Tensor result)
    {
        Broadcasting.Binary<MultiplyOperator>(a, b, result);
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

    public static void Sum(Tensor a, Tensor result)
    {
        Aggregation.Aggregate<AddOperator>(a, result);
    }

    public static void Sum(Tensor a, HashSet<int> axis, Tensor result)
    {
        Aggregation.Aggregate<AddOperator>(a, axis, result);
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

        var rowCount = a.PrevDimension;
        var colCount = a.LastDimension;
        var strides = Batching.GetBatchStrides(a, bT, ..^2);
        if (strides.BatchCount * rowCount > 64 && colCount > 64)
        {
            ParallelOptions options = new();
            Parallel.ForEach(GetMatMulAtoms(strides, a, bT, result), options, MatMulRow);
        }
        else
        {
            foreach (var atom in GetMatMulAtoms(strides, a, bT, result))
            {
                MatMulRow(atom);
            }
        }
    }

    public static void MatMulTransposedST(Tensor a, Tensor bT, Tensor result)
    {
        a.EnsureHasUpdatedValues();
        bT.EnsureHasUpdatedValues();

        var strides = Batching.GetBatchStrides(a, bT, ..^2);

        foreach (var atom in GetMatMulAtoms(strides, a, bT, result))
        {
            MatMulRow(atom);
        }
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
        var rowCount = a.PrevDimension;
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
        var rowSize = inputs.A.LastDimension;
        var rowCount = inputs.A.PrevDimension;
        var colCount = inputs.B.PrevDimension;

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
                var dot = Primitives.Dot(row, col);
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
    
    internal static Tensor[] NotSupported(Tensor grad, Tensor self)
    {
        throw new NotSupportedException("Operations is not supported");
    }
}