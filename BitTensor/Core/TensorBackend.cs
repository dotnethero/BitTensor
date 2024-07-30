using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using BitTensor.Abstractions;
using BitTensor.Internals;
using BitTensor.Operators;
using BitTensor.Playground;

namespace BitTensor.Core;

internal readonly unsafe struct TensorBackend : ITensorBackend<Tensor>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteNegate(Tensor a, Tensor result)
    {
        TensorPrimitives.Negate(a.Values, result.Data);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteSigmoid(Tensor a, Tensor result)
    {
        TensorPrimitives.Sigmoid(a.Values, result.Data);
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteTanh(Tensor a, Tensor result)
    {
        TensorPrimitives.Tanh(a.Values, result.Data);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteAdd(Tensor a, float b, Tensor result)
    {
        TensorPrimitives.Add(a.Values, b, result.Data);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteAdd(Tensor a, Tensor b, Tensor result)
    {
        Broadcasting.Binary<AddOperator>(a, b, result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteMultiply(Tensor a, float b, Tensor result)
    {
        TensorPrimitives.Multiply(a.Values, b, result.Data);
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteMultiply(Tensor a, Tensor b, Tensor result)
    {
        Broadcasting.Binary<MultiplyOperator>(a, b, result);
    }

    public static void ExecuteOuter(Tensor a, Tensor b, Tensor result)
    {
        var span = result.Data;

        fixed (float* ap = a.Values)
        {
            for (var i = 0; i < a.Size; i++)
            {
                TensorPrimitives.Multiply(b.Values, ap[i], span[(i * b.Size)..]);
            }
        }
    }

    public static void ExecuteSum(Tensor a, Tensor result)
    {
        Aggregation.Aggregate<AddOperator>(a, result);
    }

    public static void ExecuteSum(Tensor a, HashSet<int> axis, Tensor result)
    {
        Aggregation.Aggregate<AddOperator>(a, axis, result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteBroadcast(Tensor a, Tensor result) // TODO: support axis
    {
        Array.Fill(result.Data, a.Values[0]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteDot(Tensor a, Tensor b, Tensor result)
    {
        result.Data[0] = TensorPrimitives.Dot(a.Values, b.Values);
    }

    public static void ExecuteMatVecMul(Tensor a, Tensor b, Tensor result)
    {
        var (batchCount, rowCount, rowSize) = Shapes.GetBatchRowsColumns(a.Shape);

        a.EnsureHasUpdatedValues();
        b.EnsureHasUpdatedValues();

        var col = b.Data;

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

    public static void ExecuteVecMatMul(Tensor a, Tensor bT, Tensor result)
    {
        var (batchCount, colCount, rowSize) = Shapes.GetBatchRowsColumns(bT.Shape);

        a.EnsureHasUpdatedValues();
        bT.EnsureHasUpdatedValues();

        var row = a.Data;

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

    public static void ExecuteMatMulTransposed(Tensor a, Tensor bT, Tensor result)
    {
        a.EnsureHasUpdatedValues();
        bT.EnsureHasUpdatedValues();

        var rowCount = a.PrevDimension;
        var colCount = a.LastDimension;
        var strides = Batching.GetBatchStrides(a, bT, ..^2);
        if (strides.BatchCount * rowCount > 64 && colCount > 64)
        {
            ParallelOptions options = new();
            Parallel.ForEach(Batching.GetMatMulRows(strides, a, bT, result), options, MatMulRow);
        }
        else
        {
            foreach (var atom in Batching.GetMatMulRows(strides, a, bT, result))
            {
                MatMulRow(atom);
            }
        }
    }

    public static void ExecuteMatMulTransposedST(Tensor a, Tensor bT, Tensor result)
    {
        a.EnsureHasUpdatedValues();
        bT.EnsureHasUpdatedValues();

        var strides = Batching.GetBatchStrides(a, bT, ..^2);

        foreach (var atom in Batching.GetMatMulRows(strides, a, bT, result))
        {
            MatMulRow(atom);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MatMulRow(MatMulRow<Tensor> inputs)
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
}