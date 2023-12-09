using System.Diagnostics.CodeAnalysis;

namespace BitTensor.Core;

[SuppressMessage("ReSharper", "ForCanBeConvertedToForeach")]
[SuppressMessage("ReSharper", "LoopCanBeConvertedToQuery")]
internal static class Shapes
{
    public static int Product(this IEnumerable<int> items)
    {
        var result = 1;

        foreach (var item in items)
        {
            result *= item;
        }

        return result;
    }

    public static int[] GetStrides(this IEnumerable<int> shape)
    {
        var items = shape.ToArray();
        if (items.Length == 0)
            return Array.Empty<int>();

        var dims = items.Length;
        var strides = new int[dims];

        strides[^1] = 1;

        for (var i = dims - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * items[i + 1];
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
