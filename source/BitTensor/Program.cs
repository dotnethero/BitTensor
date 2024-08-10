using System.Diagnostics;
using BitTensor.CUDA;
using BitTensor.CUDA.Models;
using BitTensor.CUDA.Wrappers;

namespace BitTensor;

internal class Program
{
    public static void Main()
    {
        Test_linear_module();
    }

    public static void Test_linear_module()
    {
        const int inputCount = 400;
        const int hiddenCount = 100;
        const int outputCount = 20;
        const int batchSize = 50;

        using var context = new CuTensorContext();
        using var x = CuTensor.Random.Uniform([batchSize, inputCount]).CreateNode(context);
        using var d = CuTensor.Random.Uniform([batchSize, outputCount]).CreateNode(context);

        var model = Model.Sequential(
        [
            new LinearLayer(context, inputCount, hiddenCount, a => a),
            new LinearLayer(context, hiddenCount, outputCount, a => a)
        ]);

        var compilation = model.Compile(x, d);
        var sw = Stopwatch.StartNew();
        model.Fit(compilation, lr: 1e-5f, epochs: 1000, trace: true);
        Console.WriteLine(sw.Elapsed); // cached plans: 00:00:00.410
    }
}