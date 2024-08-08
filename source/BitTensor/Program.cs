using System.Diagnostics;
using BitTensor.CUDA;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Units;

namespace BitTensor;

internal class Program
{
    public static void Main()
    {
        Test_linear_module();
    }

    public static void Test_linear_module()
    {
        const int inputCount = 3;
        const int outputCount = 1;
        const int batchSize = 5;
        const int dataDimension = 1;

        using var x = CuTensor.Random.Uniform([batchSize, inputCount]).ToNode();
        using var d = CuTensor.Random.Uniform([batchSize, outputCount]).ToNode();

        var model = Model.Sequential(
        [
            new LinearLayer(x.Shape[dataDimension], d.Shape[dataDimension], a => a)
        ]);

        var compilation = model.Compile(x, d);
        var sw = Stopwatch.StartNew();
        model.Fit(compilation, lr: 0.001f, epochs: 10000, trace: true);
        Console.WriteLine(sw.Elapsed);
    }
}