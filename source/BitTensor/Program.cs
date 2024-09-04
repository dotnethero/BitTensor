using System.Diagnostics;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Models;
using BitTensor.CUDA.Models.Layers;

namespace BitTensor;

internal static class Program
{
    public static void Main()
    {
        Environment.SetEnvironmentVariable("CUDNN_LOGDEST_DBG", "stderr");
        Environment.SetEnvironmentVariable("CUDNN_LOGLEVEL_DBG", "2");
        Test_MNIST();
    }

    private static void Test_MNIST()
    {
        var trainImages = MNIST.ReadImages(@"C:\Projects\BitTensor\mnist\train-images.idx3-ubyte");
        var trainLabels = MNIST.ReadLabels(@"C:\Projects\BitTensor\mnist\train-labels.idx1-ubyte");
        
        var testImages = MNIST.ReadImages(@"C:\Projects\BitTensor\mnist\t10k-images.idx3-ubyte");
        var testLabels = MNIST.ReadLabels(@"C:\Projects\BitTensor\mnist\t10k-labels.idx1-ubyte");

        const int batchSize = 2048;
        const int inputCount = 28 * 28;
        const int hiddenCount = 512;
        const int outputCount = 10;

        using var context = CudaContext.CreateDefault();

        var model = Model.Create(
        [
            new Flatten<float>(context),
            //new Linear(context, inputCount, hiddenCount, Activation.ReLU(alpha: 0.1f)),
            new LinearRelu(context, inputCount, hiddenCount, alpha: 0.1f),
            new Linear(context, hiddenCount, outputCount, Activation.Softmax(CudaBackend.cuTENSOR))
        ]);

        // train:
        var trainer = Model.Compile(model, Loss.CrossEntropy, trainImages, trainLabels, batchSize);
        var timer = Stopwatch.StartNew();
        trainer.Fit(lr: 5e-3f, epochs: 10, trace: true);
        timer.Stop();

        // evaluate:
        // CuDebug.WriteLine(labels);
        // CuDebug.WriteLine(output);

        // debug:
        // CuDebug.WriteExpressionTree(trainer.Loss);

        Console.WriteLine(timer.Elapsed);

        // 5e-3f, epochs: 50, trace: true
        // 1.276
        // 00:00:02.8285282
        // 38 operation plans disposed
        // 31 arrays (7694 kB) disposed
    }
}