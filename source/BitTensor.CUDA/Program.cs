namespace BitTensor.CUDA;

internal class Program
{
    public static void Main()
    {
        using var a = CuTensor.Random.Uniform([2, 3, 4]);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(CuTensor.Transpose(a, [0, 1, 2]));
        CuDebug.WriteLine(CuTensor.Transpose(a, [0, 2, 1]));
        CuDebug.WriteLine(CuTensor.Transpose(a, [1, 2, 0]));

        CuDebug.WriteLine(CuTensor.Sum(a, [0]));
        CuDebug.WriteLine(CuTensor.Sum(a, [0, 1]));
        CuDebug.WriteLine(CuTensor.Sum(a, [0, 1, 2]));
        CuDebug.WriteLine(CuTensor.Sum(a));
    }
}