namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        using var a = CuTensor.Random.Uniform([3, 4]);
        using var b = CuTensor.Random.Uniform([   4]);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(b);
        CuDebug.WriteLine(b + a);

        return;

        Console.WriteLine(new string('=', 40));
    }
}