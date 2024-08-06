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

        BenchCuBLAS();
        BenchCuBLAS();
        BenchCuBLAS();

        BenchCuTensor();
        BenchCuTensor();
        BenchCuTensor();

        CuAsserts.ValuesAreEqual(z2, z1);

        return;

        double GFLOPS(TimeSpan elapsed) => (2 / elapsed.TotalSeconds * M * N * K) / 1e9 * B;
        
        void BenchCuBLAS()
        {
            var sw = Stopwatch.StartNew();
            CuBLAS.Multiply(a, b, z1);
            cudaRT.cudaDeviceSynchronize();

            var flops = GFLOPS(sw.Elapsed);
            Console.WriteLine($"cuBLAS: {sw.Elapsed}, {flops} GFLOPs");
        }

        void BenchCuTensor()
        {
            var sw = Stopwatch.StartNew();
            CuBLAS.Multiply(a, b, z2);
            cudaRT.cudaDeviceSynchronize();

            var flops = GFLOPS(sw.Elapsed);
            Console.WriteLine($"cuTENSOR: {sw.Elapsed}, {flops} GFLOPs");
        }
    }
}