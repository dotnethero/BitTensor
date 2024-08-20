﻿using System.Diagnostics;
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

        using var context = CudaContext.CreateDefault();

        var images = context.CreateNode<float>([batchSize, inputCount]);
        var labels = context.CreateNode<float>([batchSize, outputCount]);

        var indexes = Enumerable.Range(0, batchSize).ToArray();

        Random.Shared.Shuffle(indexes);
        images.LoadBatches(trainImages, indexes);
        labels.LoadBatches(trainLabels, indexes);

        var model = Model.Sequential(
        [
            new LinearLayer(context, inputCount, hiddenCount, new ReLU(0.1f)),
            new LinearLayer(context, hiddenCount, outputCount, Ops.Softmax)
        ]);

        // train
        var sw = Stopwatch.StartNew();
        var compilation = model.Compile(images, labels, Loss.CrossEntropy);
        model.Fit(compilation, lr: 3e-4f, epochs: 3000, trace: true);
        Console.WriteLine(sw.Elapsed); // 00:00:02.419

        // evaluate
        var output = model.Compute(images);
        output.EnsureHasUpdatedValues();

        // CuDebug.WriteLine(labels);
        // CuDebug.WriteLine(output);
    }

    private static void Test_linear_module()
    {
        const int inputCount = 400;
        const int hiddenCount = 1000;
        const int outputCount = 20;
        const int batchSize = 50;

        using var context = CudaContext.CreateDefault();

        var x = context.cuRAND.Normal([batchSize, inputCount]).AsNode(context);
        var d = context.cuRAND.Normal([batchSize, outputCount]).AsNode(context);

        var model = Model.Sequential(
        [
            new LinearLayer(context, inputCount, hiddenCount, new ReLU()),
            new LinearLayer(context, hiddenCount, outputCount, new Identity())
        ]);

        // train
        var sw = Stopwatch.StartNew();
        var compilation = model.Compile(x, d, Loss.MSE);
        model.Fit(compilation, lr: 1e-6f, epochs: 1000, trace: true);
        Console.WriteLine(sw.Elapsed); // 00:00:01.572

        // evaluate
        var output = model.Compute(x);
        var diff = Ops.Sum(output - d, [1]);
        diff.EnsureHasUpdatedValues();
        CuDebug.WriteLine(diff);
    }
    
    private static void Compare_softmax_gradients()
    {
        using var context = CudaContext.CreateDefault();

        var logits = context.Allocate<float>([2, 2], [0, 0, 0, 1]).AsNode(context);

        var x1 = Ops.Softmax(logits);
        var x2 = Ops.SoftmaxRaw(logits);

        CuDebug.WriteLine(x1);
        CuDebug.WriteLine(x2);
        CuDebug.WriteLine(Ops.DotProduct(x1, x1).GetGradients().By(logits));
        CuDebug.WriteLine(Ops.DotProduct(x2, x2).GetGradients().By(logits));
    }

    private static void Test_transpose()
    {
        using var context = CudaContext.CreateDefault();

        var a = context.cuRAND.Normal([2, 3, 4]).AsNode(context);
        var b = a.Transpose([1, 2, 0]);
        var c = b.Transpose([1, 2, 0]);
        var grads = Ops.Sum(c).GetGradients();

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(b);
        CuDebug.WriteLine(c);
    }
}