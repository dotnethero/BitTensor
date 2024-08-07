namespace BitTensor.CUDA;

internal class Program
{
    public static void Main()
    {
        using var a = new CuTensor([2, 3], [1, 2, 3, 4, 5, 6]);
        using var b = new CuTensor([3], [2, 1, 0]);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(b);
        CuDebug.WriteLine(a * b);
    }
}