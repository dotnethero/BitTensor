using System.Numerics;
using System.Runtime.InteropServices;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Wrappers;

internal sealed unsafe class CuTensorDescriptor<T> : IDisposable where T : IFloatingPoint<T>
{
    internal readonly cutensorTensorDescriptor* Descriptor;
    internal readonly cutensorOperator_t Transformation;
    internal readonly uint ModesNumber;
    internal readonly int* Modes;
    internal readonly long* Extents;
    internal readonly long* Strides;

    public CuTensorDescriptor(CuTensorContext context, Operand operand) : this(context, operand, operand.Shape.GetOrdinaryModes())
    {
    }

    public CuTensorDescriptor(CuTensorContext context, Operand operand, int[] modes)
    {
        cutensorTensorDescriptor* descriptor;

        var shape = operand.Shape;

        ModesNumber = (uint) shape.Dimensions;
        Modes = Allocate<int>(shape.Dimensions);
        Extents = Allocate<long>(shape.Dimensions);
        Strides = Allocate<long>(shape.Dimensions);

        for (var i = 0; i < shape.Dimensions; ++i)
        {
            Modes[i] = modes[i];
            Extents[i] = shape.Extents[i];
            Strides[i] = shape.Strides[i];
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
        Transformation = operand.Transformation;
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