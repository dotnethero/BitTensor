﻿using System.Diagnostics;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Graph.Epilogues;
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

        const int batchSize = 2048;
        const int inputCount = 28 * 28;
        const int hiddenCount = 512;
        const int outputCount = 10;

        using var context = CudaContext.CreateDefault();

        var model = Model.Create(
        [
            new Flatten<float>(context),
            new LinearLayer(context, inputCount, hiddenCount, new LeakyReluEpilogue(0.1f)),
            new LinearLayer(context, hiddenCount, outputCount, Ops.Softmax)
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
        // 1.484
        // 00:01:43.6200928
        // 39 operation plans disposed
        // 31 arrays (6672 kB) disposed
    }
}