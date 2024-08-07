﻿namespace BitTensor.Abstractions;

public static class ArrayExtensions
{
    public static float[] Collect2D(this float[][] items) => 
        items.Collect().ToArray();

    public static float[] Collect3D(this float[][][] items) =>
        items.Collect().Collect().ToArray();

    public static IEnumerable<T> Collect<T>(this IEnumerable<IEnumerable<T>> items) => 
        items.SelectMany(x => x);
    
    public static bool AllElementsAreUnique(this int[] items) => 
        items.ToHashSet().Count == items.Length;
    
    public static string ToText(this int[] items) => 
        $"({string.Join(",", items)})";
    
}