using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BitTensor.Core;
using BitTensor.Models;
using BitTensor.Units;

// | Method                | BatchSize | InputCount | OutputCount | Mean      | Error      | StdDev    | Median    |
// |---------------------- |---------- |----------- |------------ |----------:|-----------:|----------:|----------:|
// | Module_performace     | 1         | 10         | 10          |  2.186 ms |  0.2426 ms | 0.0375 ms |  2.169 ms |
// | Module_performace_inv | 1         | 10         | 10          |  2.164 ms |  0.1071 ms | 0.0166 ms |  2.162 ms |
// | Module_performace     | 1         | 10         | 100         |  7.726 ms | 20.0502 ms | 5.2070 ms |  9.762 ms |
// | Module_performace_inv | 1         | 10         | 100         | 12.653 ms |  3.5290 ms | 0.5461 ms | 12.657 ms |
// | Module_performace     | 1         | 100        | 10          |  8.827 ms |  2.8186 ms | 0.7320 ms |  8.621 ms |
// | Module_performace_inv | 1         | 100        | 10          |  4.563 ms | 11.9159 ms | 3.0945 ms |  3.071 ms |
// | Module_performace     | 1         | 100        | 100         | 10.879 ms | 17.6473 ms | 2.7309 ms |  9.831 ms |
// | Module_performace_inv | 1         | 100        | 100         |  8.453 ms |  7.6615 ms | 1.9897 ms |  8.505 ms |
// | Module_performace     | 1         | 1000       | 10          | 12.789 ms |  4.3549 ms | 1.1310 ms | 13.309 ms |
// | Module_performace     | 1         | 1000       | 10          | 12.438 ms | 12.6800 ms | 3.2930 ms | 11.812 ms |
// | Module_performace_inv | 1         | 1000       | 10          |  7.983 ms |  6.0814 ms | 1.5793 ms |  7.797 ms |
// | Module_performace_inv | 1         | 1000       | 10          |  7.434 ms |  1.0046 ms | 0.1555 ms |  7.462 ms |
// | Module_performace     | 1         | 1000       | 100         | 48.155 ms |  4.1411 ms | 1.0754 ms | 48.514 ms |
// | Module_performace     | 1         | 1000       | 100         | 47.610 ms |  2.5052 ms | 0.6506 ms | 47.347 ms |
// | Module_performace_inv | 1         | 1000       | 100         | 33.900 ms |  0.9775 ms | 0.2539 ms | 33.846 ms |
// | Module_performace_inv | 1         | 1000       | 100         | 33.533 ms |  0.5329 ms | 0.0825 ms | 33.561 ms |
// | Module_performace     | 10        | 10         | 10          |  5.915 ms | 20.3782 ms | 5.2922 ms |  4.794 ms |
// | Module_performace_inv | 10        | 10         | 10          |        NA |         NA |        NA |        NA |
// | Module_performace     | 10        | 10         | 100         |  4.029 ms |  0.1772 ms | 0.0274 ms |  4.018 ms |
// | Module_performace_inv | 10        | 10         | 100         |        NA |         NA |        NA |        NA |
// | Module_performace     | 10        | 100        | 10          |  7.709 ms | 13.6874 ms | 2.1181 ms |  8.047 ms |
// | Module_performace_inv | 10        | 100        | 10          |        NA |         NA |        NA |        NA |
// | Module_performace     | 10        | 100        | 100         | 14.155 ms |  0.9407 ms | 0.2443 ms | 14.085 ms |
// | Module_performace_inv | 10        | 100        | 100         |        NA |         NA |        NA |        NA |
// | Module_performace     | 10        | 1000       | 10          | 14.163 ms | 10.1089 ms | 2.6253 ms | 13.857 ms |
// | Module_performace     | 10        | 1000       | 10          | 12.856 ms |  4.2499 ms | 1.1037 ms | 13.029 ms |
// | Module_performace_inv | 10        | 1000       | 10          |        NA |         NA |        NA |        NA |
// | Module_performace_inv | 10        | 1000       | 10          |        NA |         NA |        NA |        NA |
// | Module_performace     | 10        | 1000       | 100         | 80.014 ms |  2.7641 ms | 0.7178 ms | 80.349 ms |
// | Module_performace     | 10        | 1000       | 100         | 80.061 ms |  2.9552 ms | 0.4573 ms | 80.172 ms |
// | Module_performace_inv | 10        | 1000       | 100         |        NA |         NA |        NA |        NA |
// | Module_performace_inv | 10        | 1000       | 100         |        NA |         NA |        NA |        NA |

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

        var model = Model.Sequential(
        [
            new LinearLayerInv(x.Shape[batchDimension], d.Shape[batchDimension], Tensor.Sigmoid)
        ]);

        var compilation = model.Compile(x, d);
        model.Fit(compilation, lr: 0.001f, epochs: 100);
    }
}
