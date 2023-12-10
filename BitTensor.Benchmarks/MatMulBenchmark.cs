using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BitTensor.Core;

namespace BitTensor.Benchmarks;

[SimpleJob(RunStrategy.Throughput, launchCount: 1, warmupCount: 3, iterationCount: 5, invocationCount: 100000)]
public class MatMulBenchmark
{
    private readonly Tensor x;
    private readonly Tensor y;
    private readonly Tensor z;
    private readonly Tensor yT;

    [Params(6, 20)]
    public int N { get; set; }

    public MatMulBenchmark()
    {
        x = Tensor.Random.Uniform([N, N]);
        y = Tensor.Random.Uniform([N, N]);
        yT = Tensor.Random.Uniform([N, N]);
        z = Tensor.Matmul(x, y);
    }

    [Benchmark]
    public void MatMul_cheat()
    {
        Ops.MatMulTransposed(x, y, z.Data); // only last operation
    }
    
    [Benchmark]
    public void MatMul_just_forward()
    {
        z.Forward!.Invoke(z); // has cached matrix
    }

    [Benchmark]
    public void MatMul_low_level()
    {
        var m = Ops.GetTransposeMatrix(y, [1, 0]);
        Ops.ApplyTransposeMatrix(y.Data, m, yT.Data);
        Ops.MatMulTransposed(x, yT, z.Data);
    }
    
    [Benchmark]
    public void MatMul_realistic()
    {
        x.Invalidate();
        y.Invalidate();
        z.EnsureHasUpdatedValues();
    }

    [Benchmark]
    public void MatMul_allocate()
    {
        x.Dependents.Clear(); // untracked
        y.Dependents.Clear();

        var w = Tensor.Matmul(x, y);
        w.EnsureHasUpdatedValues();
    }

    // TODO: create allocation low tensor?
}
