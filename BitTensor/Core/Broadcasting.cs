using System.Runtime.CompilerServices;
using BitTensor.Abstractions;

namespace BitTensor.Core;

internal class Broadcasting
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Binary<TOperator>(Tensor a, Tensor b, Tensor result) where TOperator : IBinaryOperator<float>
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
        var r_span = result.Data;
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
                TOperator.Execute(aslice, bslice, rslice);
            }
            else // vectorize same part
            {
                var aslice = a_span.Slice(ai * vstride, vstride);
                var bslice = b_span.Slice(bi * vstride, vstride);
                var rslice = r_span.Slice(ri * vstride, vstride);
                TOperator.Execute(aslice, bslice, rslice);
            }
        }
    }
}