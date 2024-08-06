// ReSharper disable AccessToDisposedClosure

namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        using var a = CuTensor.Random.Uniform([3, 4]);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(CuTensor.Sum(a, []));
        CuDebug.WriteLine(CuTensor.Sum(a, [0]));
        CuDebug.WriteLine(CuTensor.Sum(a, [1]));
        CuDebug.WriteLine(CuTensor.Sum(a, [0, 1]));
        CuDebug.WriteLine(CuTensor.Sum(a));
    }
}