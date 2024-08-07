namespace BitTensor.CUDA;

internal class Program
{
    public static void Main()
    {
        using var a = new CuTensor([], [99]);
        using var b = new CuTensor([3], [99, 98, 97]);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(CuTensor.Broadcast(a, [2, 3]));

        CuDebug.WriteLine(b);
        CuDebug.WriteLine(CuTensor.Broadcast(b, [3, 3]));
    }
}