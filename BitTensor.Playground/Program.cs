using BitTensor.Core;
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

        using var a = allocator.Create([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6]]);
        
        using var b = allocator.Create([
            [1, 2],
            [4, 1],
            [4, 1]]);

        using var x = CuTensor.Sum(a);
        using var y = CuTensor.Sum(a, [0]);
        using var z = CuTensor.MatMul(a, b);

        Console.WriteLine(ToHost(a).ToDataString());
        Console.WriteLine(ToHost(b).ToDataString());
        Console.WriteLine(ToHost(z).ToDataString());
    }

    private static Tensor ToHost(CuTensor c)
    {
        var data = new float[c.Size];

        c.CopyToHost(data);

        return Tensor.FromArray(c.Shape, data);
    }
}