using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BitTensor.Core;

namespace BitTensor.Benchmarks;

[SimpleJob(RunStrategy.Throughput, launchCount: 1, warmupCount: 3, iterationCount: 5)]
public class MatMulBenchmark
{
    private Tensor x;
    private Tensor y;
    private Tensor z;

    [Params(16, 64, 256, 1024)]
    public int Size { get; set; }
    
    [Params(1, 4, 16, 64)]
    public int Batches { get; set; }

    [GlobalSetup]
    public void GlobalSetup()
    {
        x = Tensor.Random.Uniform([Batches, Size, Size]);
        y = Tensor.Random.Uniform([Batches, Size, Size]);
        z = Tensor.Matmul(x, y.T);
        Console.WriteLine(z);
    }
    
    [Benchmark]
    public void MatMul()
    {
        x.Invalidate();
        y.Invalidate();
        z.EnsureHasUpdatedValues();
    }
}
