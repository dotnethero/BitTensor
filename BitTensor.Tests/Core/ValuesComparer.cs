// ReSharper disable once CheckNamespace

namespace BitTensor.Core.Tests;

class ValuesComparer(float tolerance = 0.000001f) : System.Collections.IComparer
{
    public int Compare(object? x, object? y) =>
        x is float a &&
        y is float b
            ? Compare(a, b)
            : throw new ArgumentException();

    private int Compare(float a, float b) =>
        Math.Abs(a - b) < tolerance
            ? 0 
            : Math.Sign(a - b);
}