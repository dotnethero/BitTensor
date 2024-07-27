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

        using var a = allocator.Create([1, 2, 3]);
        using var b = allocator.Create([3, 4, 5]);
        using var c = (a * b) + 10;
        using var d = (a + b) * 10;
        using var e = CuTensor.Sum(2 * a + b);

        var gradients = Auto.Grad(e);

        Console.WriteLine(ToHost(a).ToDataString());
        Console.WriteLine(ToHost(b).ToDataString());
        Console.WriteLine(ToHost(c).ToDataString());
        Console.WriteLine(ToHost(d).ToDataString());

        Console.WriteLine(ToHost(gradients.By(a)).ToDataString());
        Console.WriteLine(ToHost(gradients.By(b)).ToDataString());
    }

    private static Tensor ToHost(CuTensor c)
    {
        var data = new float[c.Size];

        c.CopyToHost(data);

        return Tensor.FromArray(c.Shape, data);
    }
}