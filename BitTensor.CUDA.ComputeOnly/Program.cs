namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        using var a = new CuTensor([3, 4]);
        using var b = new CuTensor([1, 4]);
        using var c = new CuTensor([4, 5]);

        CuDebug.WriteLine(a + b);
        CuDebug.WriteLine(a * c);
    }
}