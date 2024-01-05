using System.Runtime.CompilerServices;

namespace BitTensor.Core;

internal unsafe class Aggregation
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Aggregate<TOperator>(Tensor a, HashSet<int> axis, Tensor result) where TOperator : IAggregateOperator<float>
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
            Aggregate<TOperator>(a, result);
            return;
        }

        a.EnsureHasUpdatedValues();
        
        if (right == axis.Count)
        {
            ReduceRight<TOperator>(a.Data, a.Shape, right, result.Data);
            return;
        }
        
        if (left == axis.Count)
        {
            ReduceLeft<TOperator>(a.Data, a.Shape, left, result.Data);
            return;
        }

        // TODO: decide on the fly what is faster - left or right first
        
        if (right + left == axis.Count)
        {
            var reduced = a.Shape[..^right];
            var next = new float[reduced.Product()];
            Array.Fill(next, TOperator.Identity);
            ReduceRight<TOperator>(a.Data, a.Shape, right, next);
            ReduceLeft<TOperator>(next, reduced, left, result.Data);
            return;
        }

        var temp = a.Data;
        var shape = a.Shape;

        if (right != 0)
        {
            var reduced = shape[..^right];
            var next = new float[reduced.Product()];
            Array.Fill(next, TOperator.Identity);
            ReduceRight<TOperator>(temp, shape, right, next);
            temp = next;
            shape = reduced;
        }

        if (left != 0)
        {
            var reduced = shape[left..];
            var next = new float[reduced.Product()];
            Array.Fill(next, TOperator.Identity);
            ReduceLeft<TOperator>(temp, shape, left, next);
            temp = next;
            shape = reduced;
        }

        var axisAfterReduce = axis.Select(ax => ax - left).ToHashSet();
        AggregateNaive<TOperator>(temp, shape, axisAfterReduce, result.Data);
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Aggregate<TOperator>(Tensor a, Tensor result) where TOperator : IAggregateOperator<float>
    {
        result.Data[0] = TOperator.Aggregate(a.Values);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AggregateNaive<TOperator>(ReadOnlySpan<float> data, int[] shape, HashSet<int> axis, Span<float> result) where TOperator : IAggregateOperator<float>
    {
        result.Fill(TOperator.Identity);

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

                rp[index] = TOperator.Execute(rp[index], ap[i]);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ReduceLeft<TOperator>(ReadOnlySpan<float> data, int[] shape, int dimensions, Span<float> result) where TOperator : IAggregateOperator<float>
    {
        result.Fill(TOperator.Identity);

        var count = shape[..dimensions].Product();
        var size = shape[dimensions..].Product();

        for (var i = 0; i < count; i++)
        {
            var slice = data.Slice(i * size, size); // PERF: boundary check
            TOperator.Execute(result, slice, result);
        }
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ReduceRight<TOperator>(ReadOnlySpan<float> data, int[] shape, int dimensions, Span<float> result) where TOperator : IAggregateOperator<float>
    {
        var count = shape[..^dimensions].Product();
        var size = shape[^dimensions..].Product();

        fixed (float* rp = result)
        {
            for (var i = 0; i < count; i++)
            {
                var slice = data.Slice(i * size, size); // PERF: boundary check
                rp[i] = TOperator.Aggregate(slice);
            }
        }
    }

}