using ILGPU;

namespace BitTensor.CUDA;

using DType = float;
using DTypeView = ArrayView<float>;

internal static class CuKernels
{
    public static void Memset(Index1D i, DType value, DTypeView output)
    {
        output[i] = value;
    }
    
    public static void Negate(Index1D i, DTypeView a, DTypeView output)
    {
        output[i] = -a[i];
    }

    public static void Add(Index1D i, DTypeView a, DTypeView b, DTypeView output)
    {
        output[i] = a[i] + b[i];
    }

    public static void Add(Index1D i, DTypeView a, DType b, DTypeView output)
    {
        output[i] = a[i] + b;
    }

    public static void Mul(Index1D i, DTypeView a, DTypeView b, DTypeView output)
    {
        output[i] = a[i] * b[i];
    }
    
    public static void Mul(Index1D i, DTypeView a, DType b, DTypeView output)
    {
        output[i] = a[i] * b;
    }

    public static void BroadcastScalar(Index1D i, DTypeView a, DTypeView output)
    {
        output[i] = a[0]; // TODO: support axis
    }
    
    public static void SumToScalar(DTypeView a, DTypeView output)
    {
        var index = Grid.GlobalIndex.X;
        var memory = SharedMemory.Allocate<DType>(1);

        if (Grid.GlobalIndex.IsFirst)
        {
            output[0] = 0;
        }

        if (Group.IsFirstThread)
        {
            memory[0] = 0;
        }

        Group.Barrier();

        if (index < a.IntExtent)
            Atomic.Add(ref memory[0], a[index]);

        Group.Barrier();

        if (Group.IsFirstThread)
            Atomic.Add(ref output[0], memory[0]);
    }
}
