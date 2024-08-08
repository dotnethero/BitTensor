namespace BitTensor.CUDA;

internal class Program
{
    public static void Main()
    {
        using var a = new CuTensor([2, 3], [1, 2, 3, 4, 5, 6]);
        using var b = new CuTensor([], [2]);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(b);
        CuDebug.WriteLine(CuTensor.ElementwiseProduct(a, b));
    }
}