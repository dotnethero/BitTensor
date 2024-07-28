using BitTensor.Core;
using BitTensor.CUDA;
using ILGPU;

namespace BitTensor.Playground;

internal class Program
{
    static void Main(string[] args)
    {
        using var context = Context.CreateDefault();
        using var accelerator = context
            .GetPreferredDevice(preferCPU: false)
            .CreateAccelerator(context);

        var allocator = new CuAllocator(accelerator);

        using var a = allocator.Create([
            [1, 2, 3],
            [4, 5, 6]]);

        using var x = CuTensor.Sum(a);
        using var y = CuTensor.Sum(a, [1]);

        Console.WriteLine(ToHost(a).ToDataString());
        Console.WriteLine(ToHost(x).ToDataString());
        Console.WriteLine(ToHost(y).ToDataString());
    }

    private static Tensor ToHost(CuTensor c)
    {
        var data = new float[c.Size];

        c.CopyToHost(data);

        return Tensor.FromArray(c.Shape, data);
    }
}