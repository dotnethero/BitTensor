namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        using var a = CuTensor.Random.Uniform([3, 4]);
        using var b = CuTensor.Random.Uniform([3, 1]);
        using var c = CuTensor.Random.Uniform([1, 4]);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(b);
        CuDebug.WriteLine(a + b);

        Console.WriteLine(new string('=', 40));

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(c);
        CuDebug.WriteLine(a + c);
    }
}