using ILGPU;

namespace BitTensor.CUDA;

using DType = float;
using DTypeView = ArrayView<float>;
using DShapeView = ArrayView<int>;

internal static class CuKernels
{
    public static void Memset(Index1D i, DTypeView output, DType value)
    {
        output[i] = value;
    }
    
    public static void Negate(Index1D i, DTypeView a, DTypeView output)
    {
        output[i] = -a[i];
    }

    public static void Add(Index1D i, DTypeView a, DShapeView aStrides, DTypeView b, DShapeView bStrides, DTypeView c, DShapeView cStrides)
    {
        var aIndex = 0;
        var bIndex = 0;
        var leftover = i.X;
        var dims = cStrides.Length;

        for (var j = 0; j < dims; ++j)
        {
            var di = leftover / cStrides[j]; // dimension index
            aIndex += aStrides[j] * di;
            bIndex += bStrides[j] * di;
            leftover -= di * cStrides[j];
        }

        c[i] = a[aIndex] + b[bIndex];
    }
    
    public static void Add(Index1D i, DTypeView a, DType b, DTypeView output)
    {
        output[i] = a[i] + b;
    }

    public static void Mul(Index1D i, DTypeView a, DShapeView aStrides, DTypeView b, DShapeView bStrides, DTypeView c, DShapeView cStrides)
    {
        var aIndex = 0;
        var bIndex = 0;
        var leftover = i.X;
        var dims = cStrides.Length;

        for (var j = 0; j < dims; ++j)
        {
            var di = leftover / cStrides[j]; // dimension index
            aIndex += aStrides[j] * di;
            bIndex += bStrides[j] * di;
            leftover -= di * cStrides[j];
        }

        c[i] = a[aIndex] * b[bIndex];
    }
    
    public static void Mul(Index1D i, DTypeView a, DType b, DTypeView output)
    {
        output[i] = a[i] * b;
    }

    public static void BroadcastScalar(Index1D i, DTypeView a, DTypeView output)
    {
        output[i] = a[0]; // TODO: support axis
    }

    /// <summary>
    /// Sum array by specified axis
    /// </summary>
    /// <param name="i">Index</param>
    /// <param name="a">Input array</param>
    /// <param name="aStrides">Input strides</param>
    /// <param name="c">Reduced array</param>
    /// <param name="cStrides">Output strides prepended by zeros</param>
    public static void Sum(Index1D i, DTypeView a, DShapeView aStrides, DTypeView c, DShapeView cStrides)
    {
        var index = 0;
        var temp = i;
        var dims = aStrides.Length;
        for (var m = 0; m < dims; m++)
        {
            var dim_old = temp / aStrides[m];
            temp -= dim_old * aStrides[m];
            index += dim_old * cStrides[m];
        }

        Atomic.Add(ref c[index], a[i]);
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
