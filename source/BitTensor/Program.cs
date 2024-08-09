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
        const int inputCount = 3000;
        const int outputCount = 10;
        const int batchSize = 500;
        const int dataDimension = 1;

        using var x = CuTensor.Random.Uniform([batchSize, inputCount]).ToNode();
        using var d = CuTensor.Random.Uniform([batchSize, outputCount]).ToNode();

        var model = Model.Sequential(
        [
            new LinearLayer(x.Shape[dataDimension], d.Shape[dataDimension], a => a)
        ]);

        var compilation = model.Compile(x, d);
        var sw = Stopwatch.StartNew();
        model.Fit(compilation, lr: 0.0001f, epochs: 1000, trace: true);
        Console.WriteLine(sw.Elapsed);

        // test

        CuDebug.WriteLine(compilation.Output.Tensor - d.Tensor);
    }
}