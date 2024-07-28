using System;
using BitTensor.Abstractions;
using ILGPU;

namespace BitTensor.CUDA;

using DType = float;
using DTypeView = ArrayView<float>;
using DShapeView = ArrayView<int>;

internal readonly struct CuArray(DTypeView data, DShapeView shape)
{
    public readonly DTypeView Data = data;
    public readonly DShapeView Shape = shape;
}

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

    /// <summary>
    /// Sum array by specified axis
    /// </summary>
    /// <param name="a">Input array</param>
    /// <param name="old">Old strides</param>
    /// <param name="mod">New strides or zero if reduced</param>
    /// <param name="output">Reduced array</param>
    public static void Sum(DTypeView a, DShapeView old, DShapeView mod, DTypeView output)
    {
        var i = Grid.GlobalIndex.X;
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

        if (i < a.IntExtent)
        {
            var index = 0;
            var temp = i;
            var dims = old.Length;
            for (var m = 0; m < dims; m++)
            {
                var dim_old = temp / old[m];
                temp -= dim_old * old[m];
                index += dim_old * mod[m];
            }

            Atomic.Add(ref output[index], a[i]);
        }
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
