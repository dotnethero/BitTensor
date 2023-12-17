using System.Diagnostics.CodeAnalysis;

namespace BitTensor.Core;

[SuppressMessage("ReSharper", "ForCanBeConvertedToForeach")]
[SuppressMessage("ReSharper", "LoopCanBeConvertedToQuery")]
internal static class Shapes
{
    public static unsafe int Product(this int[] shape)
    {
        var dims = shape.Length;
        var result = 1;

        fixed (int* sh = shape)
        {
            for (var i = dims - 1; i >= 0; --i)
            {
                result *= sh[i];
            }
        }

        return result;
    }
    
    public static unsafe (int batch, int rows, int columns) GetBatchRowsColumns(int[] shape)
    {
        var dims = shape.Length;
        var batch = 1;
        var rows = 0;
        var columns = 0;

        fixed (int* sh = shape)
        {
            for (var i = dims - 3; i >= 0; --i)
            {
                batch *= sh[i];
            }
            rows = sh[dims - 2];
            columns = sh[dims - 1];
        }

        return (batch, rows, columns);
    }

    public static unsafe bool IsElementsUnique(this int[] shape)
    {
        var dims = shape.Length;

        fixed (int* sh = shape)
        {
            for (var i = 0; i < dims; ++i)
            for (var j = 0; j < i; ++j)
            {
                if (sh[i] == sh[j])
                    return false;
            }
        }

        return true;
    }


    public static unsafe int[] GetStrides(this int[] shape)
    {
        var dims = shape.Length;
        if (dims == 0)
            return [];

        var strides = new int[dims];

        fixed (int* sh = shape, st = strides)
        {
            st[dims - 1] = 1;

            for (var i = dims - 2; i >= 0; --i)
            {
                st[i] = st[i + 1] * sh[i + 1];
            }
        }

        return strides;
    }

    public static bool AreEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length)
            return false;

        for (var i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i])
                return false;
        }

        return true;
    }
    
    public static bool AreCompatible(int[] a, int[] b, out int[] shape)
    {
        var length = Math.Max(a.Length, b.Length);
        
        shape = new int[length];

        for (var i = 0; i < length; ++i)
        {
            var ai = i >= a.Length ? 1 : a[^(i+1)];
            var bi = i >= b.Length ? 1 : b[^(i+1)];
            if (ai != bi && ai != 1 && bi != 1)
                return false;

            shape[^(i+1)] = Math.Max(ai, bi);
        }

        return true;
    }
    
    public static int[] GetBroadcastedAxis(int[] inputShape, int[] resultShape)
    {
        var a_dims = inputShape.Length;
        var r_dims = resultShape.Length;
        var broadcasted = new List<int>(r_dims);

        for (var i = 1; i <= r_dims; i++)
        {
            if (i > a_dims || inputShape[^i] != resultShape[^i])
                broadcasted.Add(r_dims - i);
        }

        return broadcasted.ToArray();
    }

    public static void EnsureShapesAreEqual(int[] a, int[] b)
    {
        if (!AreEqual(a, b))
            throw new InvalidOperationException($"Shapes are not equal: {a.Serialize()} and {b.Serialize()}");
    }
    
    public static int[] EnsureShapesAreCompatible(int[] a, int[] b)
    {
        if (!AreCompatible(a, b, out var shape))
            throw new InvalidOperationException($"Shapes are not compatible: {a.Serialize()} and {b.Serialize()}");

        return shape;
    }

    public static string Serialize(this int[] items) => $"({string.Join(",", items)})";
}
