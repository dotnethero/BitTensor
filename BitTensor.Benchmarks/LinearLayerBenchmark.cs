using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BitTensor.Core;
using BitTensor.Models;
using BitTensor.Units;

namespace BitTensor.Benchmarks;

[SimpleJob(RunStrategy.Throughput, launchCount: 1, warmupCount: 3, iterationCount: 5, invocationCount: 10)]
public class LinearLayerBenchmark
{
    [Params(1, 10)]
    public int BatchSize { get; set; }

    [Params(10, 100, 1000, 1000)]
    public int InputCount { get; set; }
    
    [Params(10, 100)]
    public int OutputCount { get; set; }

    [Benchmark]
    public void Module_performace()
    {
        const int batchDimension = 1;

        var x = Tensor.Random.Normal([BatchSize, InputCount]);
        var d = Tensor.Random.Normal([BatchSize, OutputCount]);
        var model = Model.Sequential(
        [
            new LinearLayer(x.Shape[batchDimension], d.Shape[batchDimension], Tensor.Sigmoid)
        ]);

        var compilation = model.Compile(x, d);
        model.Fit(compilation, lr: 0.001f, epochs: 100);
    }

    [Benchmark]
    public void Module_performace_inv()
    {
        const int batchDimension = 0;

        var x = Tensor.Random.Normal([BatchSize, InputCount]).Transpose();
        var d = Tensor.Random.Normal([BatchSize, OutputCount]).Transpose();

        x.BatchDimension = 1;
        d.BatchDimension = 1;

        var model = Model.Sequential(
        [
            new LinearLayerInv(x.Shape[batchDimension], d.Shape[batchDimension], Tensor.Sigmoid)
        ]);

        var compilation = model.Compile(x, d);
        model.Fit(compilation, lr: 0.001f, epochs: 100);
    }
}
