using System.Diagnostics;
using BitTensor.CUDA;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Models;

namespace BitTensor;

internal class Program
{
    public static void Main()
    {
        Test_MNIST();
    }

    private static void Test_MNIST()
    {
        var trainImages = MNIST.ReadImages(@"C:\Projects\BitTensor\mnist\train-images.idx3-ubyte");
        var trainLabels = MNIST.ReadLabels(@"C:\Projects\BitTensor\mnist\train-labels.idx1-ubyte");
        
        var testImages = MNIST.ReadImages(@"C:\Projects\BitTensor\mnist\t10k-images.idx3-ubyte");
        var testLabels = MNIST.ReadLabels(@"C:\Projects\BitTensor\mnist\t10k-labels.idx1-ubyte");

        const int batchSize = 20;
        const int inputCount = 28 * 28;
        const int hiddenCount = 300;
        const int outputCount = 10;

        using var context = CuContext.CreateDefault();

        var images = context.Allocate<float>([batchSize, inputCount]).AsNode();
        var labels = context.Allocate<float>([batchSize, outputCount]).AsNode();

        var indexes = Enumerable.Range(0, batchSize).ToArray();

        Random.Shared.Shuffle(indexes);
        images.LoadBatches(trainImages, indexes);
        labels.LoadBatches(trainLabels, indexes);

        var model = Model.Sequential(
        [
            new LinearLayer(context, inputCount, hiddenCount, CuTensorNode.ReLU),
            new LinearLayer(context, hiddenCount, outputCount, CuTensorNode.Softmax)
        ]);

        // train
        var sw = Stopwatch.StartNew();
        var compilation = model.Compile(images, labels);
        model.Fit(compilation, lr: 3e-4f, epochs: 3000, trace: true);
        Console.WriteLine(sw.Elapsed); // 00:00:06.887

        // evaluate
        var output = model.Compute(images);
        output.EnsureHasUpdatedValues();
        CuDebug.WriteLine(labels.Tensor);
        CuDebug.WriteLine(output.Tensor);
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
            new LinearLayer(context, inputCount, hiddenCount, CuTensorNode.ReLU),
            new LinearLayer(context, hiddenCount, outputCount, a => a)
        ]);

        // train
        var sw = Stopwatch.StartNew();
        var compilation = model.Compile(x, d);
        model.Fit(compilation, lr: 1e-6f, epochs: 1000, trace: true);
        Console.WriteLine(sw.Elapsed); // 00:00:01.572

        // evaluate
        var output = model.Compute(x);
        var diff = CuTensorNode.Sum(output - d, [1]);
        diff.EnsureHasUpdatedValues();
        CuDebug.WriteLine(diff.Tensor);
    }
}