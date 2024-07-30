﻿using BitTensor.Core;
using BitTensor.CUDA;
using ILGPU;
using ILGPU.Runtime.Cuda;

namespace BitTensor.Playground;

internal class Program
{
    static void Main(string[] args)
    {
        using var context = Context.CreateDefault();
        
        var device = context.GetCudaDevice(0);

        using var accelerator = device.CreateCudaAccelerator(context);

        var allocator = new CuAllocator(accelerator);

        using var a = allocator.Create([[
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6]]]);
        
        using var b = allocator.Create([
            [1, 2],
            [4, 1],
            [4, 1]]);

        using var c = allocator.Create([[[0, 0, 1]]]);

        using var x = CuTensor.Sum(a);
        using var y = a + c;
        using var z = CuTensor.MatMul(a, b);

        var a_host = ToHost(a);
        var b_host = ToHost(b);
        var c_host = ToHost(c);
        var z_host = ToHost(z);
        var y_host = ToHost(CuTensor.Sum(y, [0, 2]));

        var z_test = Tensor.Matmul(a_host, b_host);
        var y_test = Tensor.Sum(a_host + c_host, [0, 2]);

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