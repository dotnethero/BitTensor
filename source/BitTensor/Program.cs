using System.Diagnostics;
using BitTensor.CUDA;
using BitTensor.CUDA.Models;

namespace BitTensor;

internal class Program
{
    public static void Main()
    {
        Test_linear_module();
    }

    public static void Test_linear_module()
    {
        const int inputCount = 200;
        const int hiddenCount = 50;
        const int outputCount = 10;
        const int batchSize = 50;

        using var x = CuTensor.Random.Uniform([batchSize, inputCount]).ToNode();
        using var d = CuTensor.Random.Uniform([batchSize, outputCount]).ToNode();

        var model = Model.Sequential(
        [
            new LinearLayer(inputCount, hiddenCount, a => a),
            new LinearLayer(hiddenCount, outputCount, a => a)
        ]);

        var compilation = model.Compile(x, d);
        var sw = Stopwatch.StartNew();
        model.Fit(compilation, lr: 0.0001f, epochs: 1000, trace: true);
        Console.WriteLine(sw.Elapsed); // 00:00:16.93
    }
}