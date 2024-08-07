namespace BitTensor.Abstractions;

internal static class Strides
{
    public static bool AreEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length)
            return false;

        for (var i = 0; i < a.Length; ++i)
        {
            if (a[i] != b[i])
                return false;
        }

        return true;
    }
}