using BitTensor.Core;
using BitTensor.Models;
using BitTensor.Units;
using System.Diagnostics;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("BitTensor.Tests")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("BitTensor.Benchmarks")]

namespace BitTensor;

public class Program
{
    public static void Main()
    {
        // Mojo_fun();
        // Two_layer_test();
        // Two_points_in_100D_to_10D();
        // Tensor_count();
        Module_performace_inv();
    }

    public static void Module_performace_inv()
    {
        const int batchDimension = 0;

        var x = Tensor.Random.Normal([10, 100]).Transpose();
        var d = Tensor.Random.Normal([10, 5]).Transpose();

        var model = Model.Sequential(
        [
            new LinearLayerInv(x.Shape[batchDimension], d.Shape[batchDimension], Tensor.Sigmoid)
        ]);

        var compilation = model.Compile(x, d);
        model.Fit(compilation, lr: 0.001f, epochs: 100);
    }

    public static void Two_points_in_100D_to_10D()
    {
        var x = Tensor.Random.Uniform([2, 100]);
        var d = Tensor.Random.Uniform([2, 10]);
        var sw = Stopwatch.StartNew();

        var model = Model.Sequential(
        [
            new LinearLayer(x.Shape[1], d.Shape[1], Tensor.Tanh)
        ]);
        var compilation = model.Compile(x, d);
        model.Fit(compilation, lr: 0.01f, epochs: 1000);
    }

    public static void Tensor_count()
    {
        var x = Tensor.Random.Normal([1000, 4]);
        var d = Tensor.Random.Normal([1000, 1]);
        var sw = Stopwatch.StartNew();

        var model = new TestModel(x.Shape[1], 7, d.Shape[1]);
        var compilation = model.Compile(x, d);
        Console.WriteLine(Tensor.MaxID);
    }

    public static void Two_layer_test()
    {
        var x = Tensor.Create([[0, 0], [0, 1], [1, 0], [1, 1]]);
        var d = Tensor.Create([0, 1, 1, 0]).Reshape([4, 1]);

        var model = new TestModel(2, 7, 1);

        var test1 = model.Compute(x).ToDataString();

        Console.WriteLine(test1);

        var compilation = model.Compile(x, d);
        model.Fit(compilation, 0.01f, epochs: 1000);
        
        var test2 = model.Compute(x).ToDataString();

        Console.WriteLine(test2);
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
