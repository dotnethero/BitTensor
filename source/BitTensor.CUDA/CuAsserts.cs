using BitTensor.Abstractions;

namespace BitTensor.CUDA;

internal static class CuAsserts
{
    public static unsafe void ValuesAreEqual(CuTensor expected, CuTensor actual, float tolerance = 1e-4f)
    {
        Shapes.EnsureAreEqual(expected.Shape, actual.Shape);

        var expects = expected.CopyToHost();
        var actuals = actual.CopyToHost();

        var size = expects.Length;

        fixed (float* ep = expects, ap = actuals)
        {
            for (var i = 0; i < size; ++i)
            {
                var diff = ep[i] - ap[i];
                if (diff > tolerance)
                {
                    throw new Exception($"Values at {i} are different: expected {expects[i]}, but {actuals[i]}");
                }
            }
        }
    }
}