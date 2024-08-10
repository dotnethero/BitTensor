using ILGPU;
using ILGPU.Runtime;

namespace BitTensor.CUDA.Kernels;

using View = ArrayView<float>;

public static class CuKernelRunner
{
    public static void Set(this Accelerator accelerator, View output, float value)
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, View, float>(CuKernelCode.Set);
        kernel(output.IntExtent, output, value);
    }

    public static void Add(this Accelerator accelerator, View a, View b, View c)
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, View, View, View>(CuKernelCode.Add);
        kernel(c.IntExtent, a, b, c);
    }

    public static void Multiply(this Accelerator accelerator, View a, View b, View c)
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, View, View, View>(CuKernelCode.Multiply);
        kernel(c.IntExtent, a, b, c);
    }
}
