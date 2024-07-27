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

        using var a = CuTensor.Create(accelerator, [1, 2, 3]);
        using var b = CuTensor.Create(accelerator, [3, 4, 5]);
        using var c = a * b * 10;
        using var d = (a + b) * (a + b);

        Console.WriteLine(ToHost(a).ToDataString());
        Console.WriteLine(ToHost(b).ToDataString());
        Console.WriteLine(ToHost(c).ToDataString());
        Console.WriteLine(ToHost(d).ToDataString());
    }

    private static Tensor ToHost(CuTensor c)
    {
        var data = new float[c.Size];

        c.CopyToHost(data);

        return Tensor.FromArray(c.Shape, data);
    }
}