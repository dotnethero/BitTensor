using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace BitTensor.Playground;

public static partial class Operations
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe void BinaryOperation<T, TOperator>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> r) 
        where TOperator : IBinaryOperator<T> 
        where T : unmanaged, IBinaryNumber<T>
    {
        var av = MemoryMarshal.Cast<T, Vector256<T>>(a);
        var bv = MemoryMarshal.Cast<T, Vector256<T>>(b);
        var rv = MemoryMarshal.Cast<T, Vector256<T>>(r);
        
        var arrSize = a.Length;
        var vecSize = av.Length;
        var simdSize = Vector256<T>.Count;

        if (vecSize > 0)
        {
            fixed (Vector256<T>* ap = av, bp = bv, rp = rv)
            {
                for (var i = 0; i < vecSize; ++i)
                {
                    rp[i] = TOperator.Invoke(ap[i], bp[i]);
                }
            }
        }

        var leftover = arrSize - vecSize * simdSize;
        if (leftover > 0)
        {
            fixed (T* ap = a, bp = b, rp = r)
            {
                for (var i = arrSize - leftover; i < arrSize; ++i)
                {
                    rp[i] = TOperator.Invoke(ap[i], bp[i]);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe T AggregateUnary<T, TOperator, TAggregate>(ReadOnlySpan<T> a) 
        where TOperator : IUnaryOperator<T> 
        where TAggregate : IBinaryOperator<T> 
        where T : unmanaged, IBinaryNumber<T>
    {
        var result = TAggregate.Identity;
        var av = MemoryMarshal.Cast<T, Vector256<T>>(a);

        var arrSize = a.Length;
        var vecSize = av.Length;
        var simdSize = Vector256<T>.Count;

        if (vecSize > 0)
        {
            var vr = Vector256.Create(TAggregate.Identity);
            fixed (Vector256<T>* ap = av)
            {
                for (var i = 0; i < vecSize; ++i)
                {
                    vr = TAggregate.Invoke(vr, TOperator.Invoke(ap[i]));
                }
            }

            for (var i = 0; i < simdSize; ++i)
            {
                result = TAggregate.Invoke(result, vr[i]);
            }
        }

        var leftover = arrSize - vecSize * simdSize;
        if (leftover > 0)
        {
            fixed (T* ap = a)
            {
                for (var i = arrSize - leftover; i < arrSize; ++i)
                {
                    result = TAggregate.Invoke(result, TOperator.Invoke(ap[i]));
                }
            }
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe T AggregateBinary<T, TOperator, TAggregate>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) 
        where TOperator : IBinaryOperator<T> 
        where TAggregate : IBinaryOperator<T> 
        where T : unmanaged, IBinaryNumber<T>
    {
        var result = TAggregate.Identity;

        var av = MemoryMarshal.Cast<T, Vector256<T>>(a);
        var bv = MemoryMarshal.Cast<T, Vector256<T>>(b);

        var arrSize = a.Length;
        var vecSize = av.Length;
        var simdSize = Vector256<T>.Count;

        if (vecSize > 0)
        {
            var vr = Vector256.Create(TAggregate.Identity);
            fixed (Vector256<T>* ap = av, bp = bv)
            {
                for (var i = 0; i < vecSize; ++i)
                {
                    vr = TAggregate.Invoke(vr, TOperator.Invoke(ap[i], bp[i]));
                }
            }

            for (var i = 0; i < simdSize; ++i)
            {
                result = TAggregate.Invoke(result, vr[i]);
            }
        }

        var leftover = arrSize - vecSize * simdSize;
        if (leftover > 0)
        {
            fixed (T* ap = a, bp = b)
            {
                for (var i = arrSize - leftover; i < arrSize; ++i)
                {
                    result = TAggregate.Invoke(result, TOperator.Invoke(ap[i], bp[i]));
                }
            }
        }

        return result;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe T AggregateTernary<T, TOperatorAggregate>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) 
        where TOperatorAggregate : ITernaryOperator<T> 
        where T : unmanaged, IBinaryNumber<T>
    {
        var result = TOperatorAggregate.Identity;

        var av = MemoryMarshal.Cast<T, Vector256<T>>(a);
        var bv = MemoryMarshal.Cast<T, Vector256<T>>(b);

        var arrSize = a.Length;
        var vecSize = av.Length;
        var simdSize = Vector256<T>.Count;

        if (vecSize > 0)
        {
            var vr = Vector256.Create(TOperatorAggregate.Identity);
            fixed (Vector256<T>* ap = av, bp = bv)
            {
                for (var i = 0; i < vecSize; ++i)
                {
                    vr = TOperatorAggregate.Invoke(ap[i], bp[i], vr);
                }
            }

            for (var i = 0; i < simdSize; ++i)
            {
                result += vr[i]; // just aggregate
            }
        }

        var leftover = arrSize - vecSize * simdSize;
        if (leftover > 0)
        {
            fixed (T* ap = a, bp = b)
            {
                for (var i = arrSize - leftover; i < arrSize; ++i)
                {
                    result = TOperatorAggregate.Invoke(ap[i], bp[i], result);
                }
            }
        }

        return result;
    }
}
