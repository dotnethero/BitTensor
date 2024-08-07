using System.Diagnostics;
using BitTensor.Core;
using BitTensor.CUDA;
using ILGPU;
using ILGPU.Runtime.Cuda;

namespace BitTensor.Playground;

internal class BenchmarkGPU
{
    public static void Run()
    {
        Bench(1024, 1024, 1024, 8, 2); // warmup
        
        Console.WriteLine();

        Bench(4096, 4096, 4096, 64, 1);

        Console.WriteLine();

        Bench(2048, 2048, 2048, 8, 8);
        Bench(2048, 2048, 2048, 64, 1);
        Bench(2048, 2048, 2048, 1, 64);

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
    }
    
    private static void Bench(int m, int n, int k, int batches, int times)
    {
        using var context = Context.CreateDefault();
        var device = context.GetCudaDevice(0);

        using var accelerator = device.CreateCudaAccelerator(context);

        var x_data = Tensor.Random.Uniform([batches, m, n]);
        var y_data = Tensor.Random.Uniform([batches, n, k]);

        using var x = new CuTensor(accelerator, x_data.Shape, x_data.Data);
        using var y = new CuTensor(accelerator, y_data.Shape, y_data.Data);
        using var z = new CuTensor(accelerator, [batches, m, k]);

        var sw = Stopwatch.StartNew();
        for (var i = 0; i < times; i++)
        {
            CuBackend.ExecuteMatMul(x, y, z);
            accelerator.Synchronize();
        }

        var s = sw.Elapsed;
        var gflops = (2 / s.TotalSeconds * m * n * k) / 1e9 * times * batches;
        Console.WriteLine($"{s} @ {m}x{n}x{k}, times={times}, batches={batches}, {gflops:0.00} GFLOP/s");
        Console.WriteLine(s);
    }
}
