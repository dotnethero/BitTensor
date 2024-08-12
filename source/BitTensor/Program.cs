using System.Diagnostics;
using BitTensor.CUDA;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Models;

namespace BitTensor;

internal class Program
{
    public static void Main()
    {
    }

    private static void Test_MNIST()
    {
        var trainImages = MNIST.ReadImages(@"C:\Projects\BitTensor\mnist\train-images.idx3-ubyte");
        var trainLabels = MNIST.ReadLabels(@"C:\Projects\BitTensor\mnist\train-labels.idx1-ubyte");
        
        var testImages = MNIST.ReadImages(@"C:\Projects\BitTensor\mnist\t10k-images.idx3-ubyte");
        var testLabels = MNIST.ReadLabels(@"C:\Projects\BitTensor\mnist\t10k-labels.idx1-ubyte");
    }

    private static void Test_linear_module()
    {
        const int inputCount = 400;
        const int hiddenCount = 1000;
        const int outputCount = 20;
        const int batchSize = 50;

        using var context = CuContext.CreateDefault();

        var x = context.Random.Normal([batchSize, inputCount]).AsNode();
        var d = context.Random.Normal([batchSize, outputCount]).AsNode();

        var model = Model.Sequential(
        [
            new LinearLayer(context, inputCount, hiddenCount, a => CuTensorNode.LeakyReLU(a, alpha: 0.01f)),
            new LinearLayer(context, hiddenCount, outputCount, a => a)
        ]);

        // train
        var sw = Stopwatch.StartNew();
        var compilation = model.Compile(x, d);
        model.Fit(compilation, lr: 1e-6f, epochs: 3000, trace: true);
        Console.WriteLine(sw.Elapsed); // 00:00:01.572

        // evaluate
        var output = model.Compute(x);
        var diff = CuTensorNode.Sum(output - d, [1]);
        diff.EnsureHasUpdatedValues();
        CuDebug.WriteLine(diff.Tensor);
    }
}