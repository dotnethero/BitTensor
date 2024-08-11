﻿using System.Diagnostics;
using BitTensor.CUDA;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Models;

namespace BitTensor;

internal class Program
{
    public static void Main()
    {
        Test_linear_module();
    }

    private static void Test_linear_module()
    {
        const int inputCount = 400;
        const int hiddenCount = 100;
        const int outputCount = 20;
        const int batchSize = 50;

        using var context = new CuContext();

        var x = context.Random.Normal([batchSize, inputCount]).AsNode();
        var d = context.Random.Normal([batchSize, outputCount]).AsNode();

        var model = Model.Sequential(
        [
            new LinearLayer(context, inputCount, hiddenCount, a => a),
            new LinearLayer(context, hiddenCount, outputCount, a => a)
        ]);

        // train
        var sw = Stopwatch.StartNew();
        var compilation = model.Compile(x, d);
        model.Fit(compilation, lr: 1e-6f, epochs: 1000, trace: true);
        Console.WriteLine(sw.Elapsed); // 00:00:00.500

        // evaluate
        var output = model.Compute(x);
        var diff = CuTensorNode.Sum(output - d, [1]);
        diff.EnsureHasUpdatedValues();
        CuDebug.WriteLine(diff.Tensor);
    }

    private static void Test_half()
    {
        using var context = new CuContext();

        var a = context.Allocate([3], [(Half)1, (Half)2, (Half)3]).AsNode();
        var b = context.Allocate([3], [(Half)4, (Half)7, (Half)2]).AsNode();
        var c = a - b;

        c.EnsureHasUpdatedValues();

        CuDebug.WriteLine(a.Tensor);
        CuDebug.WriteLine(b.Tensor);
        CuDebug.WriteLine(c.Tensor);
    }
}