using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BitTensor.Core;

// | Method              | N  | Mean        | Error     | StdDev     |
// |-------------------- |--- |------------:|----------:|-----------:|
// | MatMul_cheat        | 6  |   109.29 ns |  65.37 ns |  16.975 ns |
// | MatMul_just_forward | 6  |    98.65 ns |  26.78 ns |   6.955 ns |
// | MatMul_low_level    | 6  |   138.13 ns |  19.59 ns |   3.032 ns |
// | MatMul_realistic    | 6  |   355.07 ns |  73.32 ns |  11.346 ns |
// | MatMul_allocate     | 6  | 3,126.89 ns | 484.31 ns | 125.773 ns |
// | MatMul_cheat        | 20 |    82.74 ns |  12.22 ns |   1.891 ns |
// | MatMul_just_forward | 20 |    97.96 ns |  33.37 ns |   5.164 ns |
// | MatMul_low_level    | 20 |   133.84 ns |  36.77 ns |   9.550 ns |
// | MatMul_realistic    | 20 |   349.94 ns | 200.85 ns |  52.161 ns |
// | MatMul_allocate     | 20 | 3,156.73 ns | 659.71 ns | 171.324 ns |

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
        var w = Tensor.Matmul(x, y); // allocation overhead
        w.EnsureHasUpdatedValues();
    }

    // TODO: create allocation low tensor?
}
