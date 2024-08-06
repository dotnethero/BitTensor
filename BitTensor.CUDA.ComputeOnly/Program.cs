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
        const int B = 256;
        const int N = 128;
        const int K = 512;

        using var a = CuTensor.Random.Uniform([B, N, 1]);
        using var b = CuTensor.Random.Uniform([   1, K]);

        using var z1 = CuTensor.Allocate([B, N, K]);
        using var z2 = CuTensor.Allocate([B, N, K]);

        using var context = new CuTensorContext();
        
        using var plan1 = new CuTensorMultiplyPlan(context, a, b, z1);
        BenchAdd(() => plan1.Execute(a, b, z1), B, N, K);
        BenchAdd(() => plan1.Execute(a, b, z1), B, N, K);
        BenchAdd(() => plan1.Execute(a, b, z1), B, N, K);
            ;
        using var plan2 = new CuTensorContractionPlan(context, a, b, z2);
        BenchAdd(() => plan2.Execute(a, b, z2), B, N, K);
        BenchAdd(() => plan2.Execute(a, b, z2), B, N, K);
        BenchAdd(() => plan2.Execute(a, b, z2), B, N, K);

        CuAsserts.ValuesAreEqual(z1, z2);
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