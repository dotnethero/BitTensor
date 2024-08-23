using System.Diagnostics;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Models;
using BitTensor.CUDA.Models.Layers;
using BitTensor.CUDA.Wrappers;

namespace BitTensor;

internal class Program
{
    public static void Main()
    {
        Test1();
        Test2();

        // Test_MNIST();
    }

    private static void Test1()
    {
        using var x = new CudnnTensorDescriptor<float>(1, [3, 4]);
        using var b = new CudnnTensorDescriptor<float>(2, [3, 4]);
        using var y = new CudnnTensorDescriptor<float>(3, [3, 4]);
        using var op = new CudnnPointwiseDescriptor<float>();
        using var pw = new CudnnPointwiseOperationDescriptor<float>(op, x, b, y);
        Console.WriteLine("OK");
    }
    
    private static void Test2()
    {
        using var a = new CudnnTensorDescriptor<float>(1, [3, 4]);
        using var b = new CudnnTensorDescriptor<float>(2, [4, 5]);
        using var c = new CudnnTensorDescriptor<float>(3, [3, 5]);
        using var op = new CudnnMatMulDescriptor<float>();
        using var mm = new CudnnMatMulOperationDescriptor<float>(op, a, b, c);
        Console.WriteLine("OK");
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
            new Linear(context, inputCount, hiddenCount, Activation.ReLU(0.1f)),
            new Linear(context, hiddenCount, outputCount, Activation.Softmax)
        ]);

        // train:
        var timer = Stopwatch.StartNew();
        var trainer = Model.Compile(model, Loss.CrossEntropy, trainImages, trainLabels, batchSize);
        trainer.Fit(lr: 5e-3f, epochs: 50, trace: true);
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