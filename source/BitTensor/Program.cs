using System.Diagnostics;
using BitTensor.CUDA;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Kernels;
using BitTensor.CUDA.Models;
using ILGPU;
using ILGPU.Runtime;
using Half = System.Half;

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

        using var context = CuContext.CreateDefault();

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

    private static void Test_kernel()
    {
        using var context = CuContext.CreateDefault();

        var a = context.Allocate<float>([2_000_000_000]).AsNode();
        var b = CuTensorNode.Sigmoid(a);

        var setKernel = context.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float>(CuKernels.Set);
        setKernel(a.Size, a.Tensor.Array.Buffer.View, .8f);

        var sigmoidKernel = context.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(CuKernels.Sigmoid);
        
        var sw2 = Stopwatch.StartNew();
        b.EnsureHasUpdatedValues();
        Console.WriteLine(sw2.Elapsed); // 00:00:00.500

        var sw1 = Stopwatch.StartNew();
        sigmoidKernel(a.Size, a.Tensor.Array.Buffer.View, b.Tensor.Array.Buffer.View);
        context.Accelerator.Synchronize();
        Console.WriteLine(sw1.Elapsed); // 00:00:00.500
    }

    private static void Test_half()
    {
        using var context = CuContext.CreateDefault();

        var a = context.Allocate([3], [(Half)1, (Half)2, (Half)3]).AsNode();
        var b = context.Allocate([3], [(Half)4, (Half)7, (Half)2]).AsNode();
        var c = a - b;

        c.EnsureHasUpdatedValues();

        CuDebug.WriteLine(a.Tensor);
        CuDebug.WriteLine(b.Tensor);
        CuDebug.WriteLine(c.Tensor);
    }
}