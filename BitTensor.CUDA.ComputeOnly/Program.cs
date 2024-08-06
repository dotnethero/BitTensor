using System.Diagnostics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        const int B = 64;
        const int N = 128;
        const int M = 256;
        const int K = 512;

        using var a = CuTensor.Random.Uniform([B, N, M]);
        using var b = CuTensor.Random.Uniform([B, M, K]);

        using var z1 = CuTensor.Allocate([B, N, K]);
        using var z2 = CuTensor.Allocate([B, N, K]);

        var sw = Stopwatch.StartNew();

        CuTensor.Multiply(a, b, z1);
        cudaRT.cudaDeviceSynchronize();
        Console.WriteLine($"cuTENSOR: {sw.Elapsed}");

        sw.Restart();
        
        CuTensor.Multiply(a, b, z1);
        cudaRT.cudaDeviceSynchronize();
        Console.WriteLine($"cuTENSOR: {sw.Elapsed}");
        
        sw.Restart();
        
        CuTensor.Multiply(a, b, z1);
        cudaRT.cudaDeviceSynchronize();
        Console.WriteLine($"cuTENSOR: {sw.Elapsed}");

        sw.Restart();

        CuBLAS.Multiply(a, b, z2);
        cudaRT.cudaDeviceSynchronize();
        Console.WriteLine($"cuBLAS: {sw.Elapsed}");
        
        sw.Restart();

        CuBLAS.Multiply(a, b, z2);
        cudaRT.cudaDeviceSynchronize();
        Console.WriteLine($"cuBLAS: {sw.Elapsed}");
        
        sw.Restart();

        CuBLAS.Multiply(a, b, z2);
        cudaRT.cudaDeviceSynchronize();
        Console.WriteLine($"cuBLAS: {sw.Elapsed}");

        CuAsserts.ValuesAreEqual(z2, z1);
    }
}