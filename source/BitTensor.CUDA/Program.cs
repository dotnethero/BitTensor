namespace BitTensor.CUDA;

internal class Program
{
    public static void Main()
    {
        using var a = new CuTensor([2, 2], [1, 2, 3, 4]);
        using var b = new CuTensor([3], [0, 1, 2]);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(b);
        CuDebug.WriteLine(CuTensor.Outer(a, b));
    }
}