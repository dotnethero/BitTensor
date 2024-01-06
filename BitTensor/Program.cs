using BitTensor.Core;
using System.Diagnostics;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("BitTensor.Tests")]
[assembly: InternalsVisibleTo("BitTensor.Benchmarks")]

namespace BitTensor;

public class Program
{
    public static void Main()
    {
        Bench(1024, 1024, 1024, 8, 2); // warmup

        Console.WriteLine();

        Bench(1024, 1024, 1024, 8, 8);
        Bench(1024, 1024, 1024, 64, 1);
        Bench(1024, 1024, 1024, 1, 64);

        Console.WriteLine();
        
        Bench(512, 512, 512, 8, 8);
        Bench(512, 512, 512, 64, 1);
        Bench(512, 512, 512, 1, 64);

        Console.WriteLine();
        
        Bench(256, 256, 256, 8, 8);
        Bench(256, 256, 256, 64, 1);
        Bench(256, 256, 256, 1, 64);
        
        Console.WriteLine();
        
        Bench(128, 128, 128, 8, 8);
        Bench(128, 128, 128, 64, 1);
        Bench(128, 128, 128, 1, 64);

        Console.WriteLine();

        Bench(64, 64, 64, 8, 8);
        Bench(64, 64, 64, 64, 1);
        Bench(64, 64, 64, 1, 64);

        Console.WriteLine();
        
        Bench(32, 32, 32, 8, 8);
        Bench(32, 32, 32, 64, 1);
        Bench(32, 32, 32, 1, 64);
    }
    
    private static void Bench(int m, int n, int k, int batches, int times)
    {
        var x = Tensor.Random.Uniform([batches, m, n]);
        var y = Tensor.Random.Uniform([batches, n, k]).T;
        var z = Tensor.Empty([batches, m, k]);

        var sw = Stopwatch.StartNew();
        for (var i = 0; i < times; i++)
        {
            Ops.MatMulTransposed(x, y, z);
        }

        var s = sw.Elapsed;
        var gflops = (2 / s.TotalSeconds * m * n * k) / 1e9 * times * batches;
        Console.WriteLine($"{s} @ {m}x{n}x{k}, times={times}, batches={batches}, {gflops:0.00} GFLOP/s");
    }
}
