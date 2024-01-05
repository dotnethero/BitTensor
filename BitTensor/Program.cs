using BitTensor.Core;
using System.Diagnostics;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("BitTensor.Tests")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("BitTensor.Benchmarks")]

namespace BitTensor;

public class Program
{
    public static void Main()
    {
        Mojo_fun();
        Mojo_fun();
        Mojo_fun();
    }

    public static void Mojo_fun()
    {
        // Batch: 1 | Times: 64
        // 58.11505425282896 GFLOP/s
        // 65.23007074673644 GFLOP/s
        // 64.27012281384961 GFLOP/s

        // Batch: 64 | Times: 1
        // 30.809862242058195 GFLOP/s
        // 36.263178080156166 GFLOP/s
        // 34.75909077578663 GFLOP/s

        const int m = 1024;
        const int n = 1024;
        const int k = 1024;
        const int q = 1;

        const int times = 256;

        var x = Tensor.Random.Uniform([q, m, n]);
        var y = Tensor.Random.Uniform([q, n, k]).Transpose();
        var z = Tensor.Empty([q, m, k]);

        var sw = Stopwatch.StartNew();
        for (var i = 0; i < times; i++)
        {
            Ops.MatMulTransposed(x, y, z);
        }

        var s = sw.Elapsed;
        var gflops = (2 / s.TotalSeconds * m * n * k) / 1e9 * times * q;
        Console.WriteLine($"{gflops} GFLOP/s");
    }
}
