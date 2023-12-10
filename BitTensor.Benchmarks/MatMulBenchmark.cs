using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BitTensor.Core;

namespace BitTensor.Benchmarks;

[SimpleJob(RunStrategy.Monitoring, launchCount: 1, warmupCount: 1, iterationCount: 3, invocationCount: 10_000_000)]
public class MatMulBenchmark
{
    private readonly Tensor x;
    private readonly Tensor xf;
    private readonly Tensor y;
    private readonly Tensor yf;
    private readonly Tensor z;
    private readonly Tensor yT;

    [Params(6, 20)]
    public int N { get; set; }

    public MatMulBenchmark()
    {
        x = Tensor.Random.Uniform([N, N]);
        y = Tensor.Random.Uniform([N, N]);
        xf = Tensor.Random.Uniform([N, N]);
        yf = Tensor.Random.Uniform([N, N]);
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
        var w = Tensor.Matmul(xf, yf);
        w.EnsureHasUpdatedValues();
        xf.Dependents.Clear(); // cleanup
        yf.Dependents.Clear();
    }
}
