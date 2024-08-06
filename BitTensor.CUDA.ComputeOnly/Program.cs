using System.Diagnostics;
using System.Runtime.CompilerServices;
using BitTensor.CUDA.ComputeOnly.Plans;
using BitTensor.CUDA.ComputeOnly.Wrappers;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        const int B = 64;
        const int N = 128;
        const int K = 512;

        using var a = CuTensor.Random.Uniform([B, N, K]);
        using var b = CuTensor.Random.Uniform([B, N, K]);

        using var z = CuTensor.Allocate([B, N, K]);

        using var context = new CuTensorContext();
        
        using var plan1 = new CuTensorElementwiseMultiply(context, a, b, z);
        BenchAdd(() => plan1.Execute(a, b, z), B, N, K);
        BenchAdd(() => plan1.Execute(a, b, z), B, N, K);
        BenchAdd(() => plan1.Execute(a, b, z), B, N, K);
            ;
        using var plan2 = new CuTensorElementwiseMultiplyContraction(context, a, b, z);
        BenchAdd(() => plan2.Execute(a, b, z), B, N, K);
        BenchAdd(() => plan2.Execute(a, b, z), B, N, K);
        BenchAdd(() => plan2.Execute(a, b, z), B, N, K);
    }

    private static void BenchAdd(Action action, int B, int N, int K, [CallerArgumentExpression("action")] string actionName = "")
    {
        var sw = Stopwatch.StartNew();
        action();
        cudaRT.cudaDeviceSynchronize();

        var flops = (B * N * K / sw.Elapsed.TotalSeconds) / 1e9;
        Console.WriteLine($"{actionName}: {sw.Elapsed}, {flops} GFLOPs");
    }
}