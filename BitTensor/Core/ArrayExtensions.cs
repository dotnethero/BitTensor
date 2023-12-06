namespace BitTensor.Core;

internal static class ArrayExtensions
{
    public static float Scalar(this ReadOnlySpan<float> values) => values[0];

    public static float[] Collect2D(this float[][] items) => 
        items
            .Collect()
            .ToArray();

    public static float[] Collect3D(this float[][][] items) =>
        items
            .Collect()
            .Collect()
            .ToArray();

    private static IEnumerable<T> Collect<T>(this IEnumerable<IEnumerable<T>> items) => 
        items.SelectMany(x => x);
}