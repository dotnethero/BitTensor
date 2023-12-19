using BitTensor.Core;
using System.Diagnostics;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("BitTensor.Tests")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("BitTensor.Benchmarks")]

namespace BitTensor;

public class Program
{
    public static void Main()
    {
        // [50, 1000, 1000] @ [5, 1, 1000, 1000] = 00:00:22.4988747
        var a = Tensor.Random.Normal([50, 1000, 1000]);
        var b = Tensor.Random.Normal([5, 1, 1000, 1000]);
        var sw = Stopwatch.StartNew();
        var c = Tensor.Matmul(a, b);
        Console.WriteLine(c.Values[0]);
        Console.WriteLine(sw.Elapsed);
    }

    public static void Mojo_fun()
    {
        // 12.906805382073648 GFLOP/s
        // 41.253589387363675 GFLOP/s parallel
        // 55.90393228520536 GFLOP/s parallel x 16

        const int m = 1024;
        const int n = 1024;
        const int k = 1024;

        const int times = 100;

        var x = Tensor.Random.Uniform([m, n]);
        var y = Tensor.Random.Uniform([n, k]).Transpose();
        var z = new float[m * k];

        var sw = Stopwatch.StartNew();
        for (var i = 0; i < times; i++)
        {
            Ops.MatMulTransposed(x, y, z);
        }

        var s = sw.Elapsed;
        var gflops = (2 / s.TotalSeconds * m * n * k) / 1e9 * times;
        Console.WriteLine($"{gflops} GFLOP/s");
    }
}
