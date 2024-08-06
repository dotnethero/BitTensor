using System.Runtime.InteropServices;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

internal unsafe class CuTensorDescriptor : IDisposable
{
    internal readonly cutensorTensorDescriptor* Descriptor;
    internal readonly uint ModesNumber;
    internal readonly int* Modes;
    internal readonly long* Extents;
    internal readonly long* Strides;
    internal readonly float* Data;

    public CuTensorDescriptor(CuTensorContext context, CuTensor a) : this(context, a, a.Shape.GetModes())
    {
    }

    public CuTensorDescriptor(CuTensorContext context, CuTensor a, int[] modes)
    {
        cutensorTensorDescriptor* descriptor;

        ModesNumber = (uint) a.Dimensions;
        Modes = Allocate<int>(a.Dimensions);
        Extents = Allocate<long>(a.Dimensions);
        Strides = Allocate<long>(a.Dimensions);
        Data = a.Pointer;

        for (var i = 0; i < a.Dimensions; ++i)
        {
            Modes[i] = modes[i];
            Extents[i] = a.Shape[i];
            Strides[i] = a.Strides[i];
        }

        var status = cuTENSOR.cutensorCreateTensorDescriptor(
            context.Handle,
            &descriptor,
            ModesNumber,
            Extents,
            Strides,
            dataType: cutensorDataType_t.CUTENSOR_R_32F,
            alignmentRequirement: 128u);

        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);

        Descriptor = descriptor;
    }

    private static T* Allocate<T>(int count) where T : unmanaged
    {
        return (T*) Marshal.AllocHGlobal(count * sizeof(T));
    }

    public void Dispose()
    {
        cuTENSOR.cutensorDestroyTensorDescriptor(Descriptor);

        Marshal.FreeHGlobal((IntPtr)Modes);
        Marshal.FreeHGlobal((IntPtr)Extents);
        Marshal.FreeHGlobal((IntPtr)Strides);
    }
}