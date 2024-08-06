using System.Diagnostics;
using BitTensor.CUDA.ComputeOnly.Plans;
using BitTensor.CUDA.ComputeOnly.Wrappers;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        const int B = 1024;
        const int N = 128;
        const int M = 256;
        const int K = 512;

        using var a = CuTensor.Random.Uniform([B, N, M]);
        using var b = CuTensor.Random.Uniform([M, K]);

        using var z1 = CuTensor.Allocate([B, N, K]);
        using var z2 = CuTensor.Allocate([B, N, K]);
        using var z3 = CuTensor.Allocate([B, N, K]);

        BenchCuBLAS();
        BenchCuBLAS();
        BenchCuBLAS();

        BenchCuTensor();
        BenchCuTensor();
        BenchCuTensor();

        using var context = new CuTensorContext();
        using var plan = new CuTensorMatrixMultiplication(context, a, b, z3);

        BenchCuTensorCachedPlan();
        BenchCuTensorCachedPlan();
        BenchCuTensorCachedPlan();

        CuAsserts.ValuesAreEqual(z3, z1);

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
            CuTensor.Multiply(a, b, z2);
            cudaRT.cudaDeviceSynchronize();

            var flops = GFLOPS(sw.Elapsed);
            Console.WriteLine($"cuTENSOR: {sw.Elapsed}, {flops} GFLOPs");
        }

        void BenchCuTensorCachedPlan()
        {
            var sw = Stopwatch.StartNew();
            plan.Execute(a, b, z3);
            cudaRT.cudaDeviceSynchronize();

            var flops = GFLOPS(sw.Elapsed);
            Console.WriteLine($"cuTENSOR cached: {sw.Elapsed}, {flops} GFLOPs");
        }
    }
}