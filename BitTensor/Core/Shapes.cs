using System.Diagnostics.CodeAnalysis;

namespace BitTensor.Core;

[SuppressMessage("ReSharper", "ForCanBeConvertedToForeach")]
[SuppressMessage("ReSharper", "LoopCanBeConvertedToQuery")]
internal static class Shapes
{
    public static int Product(this int[] items)
    {
        var result = 1;
        for (var i = 0; i < items.Length; i++)
        {
            result *= items[i];
        }
        return result;
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
    
    public static void EnsureShapesAreEqual(int[] a, int[] b)
    {
        if (!AreEqual(a, b))
            throw new InvalidOperationException($"Shapes are incompatible: {a.Serialize()} and {b.Serialize()}");
    }

    public static string Serialize(this int[] items) => $"({string.Join(",", items)})";
}