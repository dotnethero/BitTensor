using System.Numerics;
using System.Runtime.InteropServices;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Wrappers;

internal sealed unsafe class CuTensorDescriptor<T> : IDisposable where T : IFloatingPoint<T>
{
    internal readonly cutensorTensorDescriptor* Descriptor;
    internal readonly uint ModesNumber;
    internal readonly int* Modes;
    internal readonly long* Extents;
    internal readonly long* Strides;

    public CuTensorDescriptor(CuTensorContext context, AbstractTensor a) : this(context, a, a.Shape.GetOrdinaryModes())
    {
    }

    public CuTensorDescriptor(CuTensorContext context, AbstractTensor a, int[] modes)
    {
        cutensorTensorDescriptor* descriptor;

        ModesNumber = (uint) a.Dimensions;
        Modes = Allocate<int>(a.Dimensions);
        Extents = Allocate<long>(a.Dimensions);
        Strides = Allocate<long>(a.Dimensions);

        for (var i = 0; i < a.Dimensions; ++i)
        {
            Modes[i] = modes[i];
            Extents[i] = a.Shape.Extents[i];
            Strides[i] = a.Shape.Strides[i];
        }

        var status = cuTENSOR.cutensorCreateTensorDescriptor(
            context.Handle,
            &descriptor,
            ModesNumber,
            Extents,
            Strides,
            Types.GetDataType<T>(),
            alignmentRequirement: 128u);

        Status.EnsureIsSuccess(status);

        Descriptor = descriptor;
    }

    private static TAny* Allocate<TAny>(int count)
        where TAny : unmanaged =>
        (TAny*) Marshal.AllocHGlobal(count * sizeof(TAny));

    public void Dispose()
    {
        cuTENSOR.cutensorDestroyTensorDescriptor(Descriptor);

        Marshal.FreeHGlobal((IntPtr)Modes);
        Marshal.FreeHGlobal((IntPtr)Extents);
        Marshal.FreeHGlobal((IntPtr)Strides);
    }
}