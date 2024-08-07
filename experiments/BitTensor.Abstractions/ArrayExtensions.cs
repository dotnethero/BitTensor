﻿namespace BitTensor.Abstractions;

public static class ArrayExtensions
{
    public static float Scalar(this ReadOnlySpan<float> values) => values[0];
    
    private static IEnumerable<T> Collect<T>(this IEnumerable<IEnumerable<T>> items) => 
        items.SelectMany(x => x);

    public static float[] Collect2D(this float[][] items) => 
        items
            .Collect()
            .ToArray();

    public static float[] Collect3D(this float[][][] items) =>
        items
            .Collect()
            .Collect()
            .ToArray();
}