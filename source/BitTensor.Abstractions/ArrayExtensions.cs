namespace BitTensor.Abstractions;

public static class ArrayExtensions
{
    public static float[] Collect2D(this float[][] items) => 
        items.Collect().ToArray();

    public static float[] Collect3D(this float[][][] items) =>
        items.Collect().Collect().ToArray();

    public static IEnumerable<T> Collect<T>(this IEnumerable<IEnumerable<T>> items) => 
        items.SelectMany(x => x);

    public static unsafe bool ArrayEqualsTo(this int[] a, ReadOnlySpan<int> b)
    {
        var alen = a.Length;
        var blen = b.Length;
        if (blen != alen)
            return false;

        fixed (int* ap = a, bp = b)
            for (var i = 0; i < alen; ++i)
            {
                if (ap[i] != bp[i])
                    return false;
            }

        return true;
    }
}