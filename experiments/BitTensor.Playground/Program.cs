using System.Diagnostics;
using BitTensor.Core;
using BitTensor.CUDA;
using BitTensor.Models;
using BitTensor.Units;
using ILGPU;
using ILGPU.Runtime.Cuda;

namespace BitTensor.Playground;

internal class Program
{
    static void Main(string[] args)
    {
        Test_linear_module();
    }

    public static void Test_linear_module()
    {
        const int inputCount = 400;
        const int hiddenCount = 100;
        const int outputCount = 20;
        const int batchSize = 50;

        var x = Tensor.Random.Uniform([batchSize, inputCount]);
        var d = Tensor.Random.Uniform([batchSize, outputCount]);

        var model = Model.Sequential(
        [
            new LinearLayer(inputCount, hiddenCount, a => a),
            new LinearLayer(hiddenCount, outputCount, a => a)
        ]);

        var compilation = model.Compile(x, d);
        var sw = Stopwatch.StartNew();
        model.Fit(compilation, lr: 1e-5f, epochs: 1000, trace: true);
        Console.WriteLine(sw.Elapsed);
    }

    private static void TestTranspose()
    {
        using var context = Context.CreateDefault();
        
        var device = context.GetCudaDevice(0);

        using var accelerator = device.CreateCudaAccelerator(context);

        var allocator = new CuAllocator(accelerator);
        
        using var a = allocator.Create([[
            [1, 2, 3],
            [1, 7, 3],
            [1, 4, 3],
            [4, 5, 6]]]);

        using var b = CuTensor.Transpose(a);

        var a_host = ToHost(a);
        var b_host = ToHost(b);

        Console.WriteLine(a_host.ToDataString());
        Console.WriteLine(b_host.ToDataString());
    }

    private static void Test()
    {
        using var context = Context.CreateDefault();
        
        var device = context.GetCudaDevice(0);

        using var accelerator = device.CreateCudaAccelerator(context);

        var allocator = new CuAllocator(accelerator);

        using var a = allocator.Create([[
            [1, 2, 3],
            [1, 7, 3],
            [1, 4, 3],
            [4, 5, 6]]]);
        
        using var b = allocator.Create([
            [1, 2],
            [4, 1],
            [4, 1]]);

        using var c = allocator.Create([[[0, 2, 1]]]);

        using var x = CuTensor.Sum(a);
        using var y = a * c;
        using var z = CuTensor.MatMul(a, b);

        var a_host = ToHost(a);
        var b_host = ToHost(b);
        var c_host = ToHost(c);
        var z_host = ToHost(z);
        var y_host = ToHost(CuTensor.Sum(y, [0]));

        var z_test = Tensor.Matmul(a_host, b_host);
        var y_test = Tensor.Sum(a_host * c_host, [0]);

        Console.WriteLine(z_host.ToDataString());
        Console.WriteLine(z_test.ToDataString());
        Console.WriteLine(y_host.ToDataString());
        Console.WriteLine(y_test.ToDataString());
    }

    private static Tensor ToHost(CuTensor c)
    {
        return Tensor.FromArray(c.Shape, c.CopyToHost());
    }
}