namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        using var a = CuTensor.Random.Uniform([3, 4]);
        using var b = CuTensor.Random.Uniform([4]);
        using var c = CuTensor.Random.Uniform([4, 5]);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(a * 2);
        CuDebug.WriteLine(b);
        CuDebug.WriteLine(a + b);
        CuDebug.WriteLine(b + a);

        return;

        Console.WriteLine(new string('=', 40));
         
        CuDebug.WriteLine(a);
        CuDebug.WriteLine(c);
        CuDebug.WriteLine(a * c);
    }
}