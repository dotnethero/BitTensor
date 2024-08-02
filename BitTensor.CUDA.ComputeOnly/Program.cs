namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        using var a = CuTensor.Random.Uniform([3, 4]);
        using var b = CuTensor.Random.Uniform([3, 4]);
        using var c = CuTensor.Random.Uniform([4, 5]);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(b);
        CuDebug.WriteLine(c);
        CuDebug.WriteLine(a + b);
        CuDebug.WriteLine(a * c);
    }
}