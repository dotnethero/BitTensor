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
        Environment.SetEnvironmentVariable("CUDNN_LOGLEVEL_DBG", "2");
        Test1();

        // Test_MNIST();
    }

    private static void Test1()
    {
        using var context = new CudnnContext();
        using var t1 = new CudnnTensorDescriptor<float>(1, [3, 4]);
        using var t2 = new CudnnTensorDescriptor<float>(2, [3, 4]);
        using var t3 = new CudnnTensorDescriptor<float>(3, [3, 4]);
        using var t4 = new CudnnTensorDescriptor<float>(4, [4, 5]);
        using var t5 = new CudnnTensorDescriptor<float>(5, [3, 5]);
        using var pwc = new CudnnPointwiseOperator<float>();
        using var pw = new CudnnPointwiseOperation<float>(pwc, t1, t2, t3);
        using var mmc = new CudnnMatMulOperator<float>();
        using var mm = new CudnnMatMulOperation<float>(mmc, t3, t4, t5);
        using var graph = new CudnnGraph(context, [pw, mm]);
        using var engine = new CudnnEngine(graph, globalIndex: 0);
        using var config = new CudnnEngineConfiguration(engine);
        var ws = config.GetWorkspaceSize();
        Console.WriteLine($"OK: {ws}");
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