using System.Diagnostics;
using BitTensor.CUDA;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Models;
using BitTensor.CUDA.Models.Layers;
using BitTensor.CUDA.Wrappers;

namespace BitTensor;

internal class Program
{
    public static void Main()
    {
        Environment.SetEnvironmentVariable("CUDNN_LOGLEVEL_DBG", "0");

        Test1();

        // Test_MNIST();
    }

    private static void Test1()
    {
        var random = new CuRandContext();

        const int BATCH = 3;
        const int INPUTS = 8;
        const int OUTPUTS = 4;

        using var inputs = random.Normal([BATCH, INPUTS]);
        using var weights = random.Normal([INPUTS, OUTPUTS]);
        using var bias = random.Normal([OUTPUTS]);
        using var temp = new CudaTensor<float>([BATCH, OUTPUTS]);
        using var outputs = new CudaTensor<float>([BATCH, OUTPUTS]);

        using var cutensor = new CuTensorContext();
        using var cuplan1 = cutensor.CreateMatMulPlan<float>(inputs.Shape, weights.Shape, temp.Shape);
        using var cuplan2 = cutensor.CreateAddPlan<float>(temp.Shape, bias.Shape, outputs.Shape);

        cuplan1.Execute(inputs, weights, outputs);
        cuplan2.Execute(outputs, bias, outputs);

        CuDebug.WriteLine(outputs);

        using var context = new CudnnContext();

        using var ti = new CudnnTensorDescriptor<float>(inputs);
        using var tw = new CudnnTensorDescriptor<float>(weights);
        using var tb = new CudnnTensorDescriptor<float>(bias);
        using var tt = new CudnnTensorDescriptor<float>(outputs.Shape, -1, isVirtual: true);
        using var to = new CudnnTensorDescriptor<float>(outputs);

        using var mmc = new CudnnMatMulOperator<float>();
        using var mm = new CudnnMatMulOperation<float>(mmc, ti, tw, tt);

        using var pwc = new CudnnPointwiseOperator<float>();
        using var pw = new CudnnPointwiseOperation<float>(pwc, tt, tb, to);

        using var graph = new CudnnGraph(context, [mm, pw]);
        using var pack = new CudnnVariantPack<float>([inputs, weights, bias, outputs]);

        context.ExecuteGraph(graph, pack);
        CuDebug.WriteLine(outputs);
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