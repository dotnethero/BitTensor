using BitTensor.CUDA;

namespace BitTensor;

internal class Program
{
    public static void Main()
    {
        using var a = new CuTensor([3], [1, 2, 3]);
        using var b = new CuTensor([3], [14, 1, 2]);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(b);
        CuDebug.WriteLine(CuTensor.DotProduct(a, b));
        CuDebug.WriteLine(a * b);
    }
}