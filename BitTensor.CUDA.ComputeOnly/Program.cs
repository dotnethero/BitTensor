namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        using var a = CuTensor.Random.Uniform([3, 4]);
        using var b = CuTensor.Random.Uniform([1, 4]);
        using var c = new CuTensor([3, 4]);
        using var d = new CuTensor([3, 4]);

        CuTensor.Contract(a, b, c, d);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(b);
        CuDebug.WriteLine(c);
        CuDebug.WriteLine(d);

        return;

        Console.WriteLine(new string('=', 40));
    }
}