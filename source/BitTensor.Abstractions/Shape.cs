using System.Collections;
using System.Runtime.CompilerServices;

namespace BitTensor.Abstractions;

[CollectionBuilder(typeof(Shape), nameof(Create))]
public sealed class Shape : IEnumerable<int>
{
    public static readonly Shape Scalar = [];

    public readonly int Dimensions;
    public readonly int ArraySize;
    public readonly int[] Extents;
    public readonly int[] Strides;
    public readonly bool IsPacked;

    public static Shape Create(ReadOnlySpan<int> extents) => new(extents);
    public static Shape Create(ReadOnlySpan<int> extents, ReadOnlySpan<int> strides) => new(extents, strides);

    private Shape(ReadOnlySpan<int> extents)
    {
        Dimensions = extents.Length;
        ArraySize = GetArraySize(extents);
        Extents = extents.ToArray();
        Strides = GetStrides(extents);
        IsPacked = true;
    }

    private Shape(ReadOnlySpan<int> extents, ReadOnlySpan<int> strides)
    {
        Dimensions = extents.Length;
        ArraySize = GetArraySize(extents);
        Extents = extents.ToArray();
        Strides = strides.ToArray();
        IsPacked = GetStrides(extents).ArrayEqualsTo(strides);
    }

    public void EnsureCanReshape(Shape shape)
    {
        if (shape.ArraySize != ArraySize)
            throw new InvalidOperationException($"Can't reshape {this} into {shape}");
    }

    public Shape Transpose() => Transpose(this.GetTransposeAxis());

    public Shape Transpose(Index[] axis)
    {
        var offsets = GetOffsets(axis).ToHashSet();
        if (offsets.Count != Dimensions)
            throw new InvalidOperationException($"Can't transpose {this} with permutation {axis.ToText()}");

        var extents = new int[Dimensions];
        var strides = new int[Dimensions];
        for (var i = 0; i < Dimensions; ++i)
        {
            extents[i] = Extents[axis[i]];
            strides[i] = Strides[axis[i]];
        }
        return Create(extents, strides);
    }
    
    public Shape Append(int size) => Create([..Extents, size]);
    
    public Shape Expand(int dimensions)
    {
        if (dimensions <= Dimensions)
            return this;
        
        var dims = Dimensions < dimensions ? dimensions : Dimensions;
        var additional = dims - Dimensions;
        var ones = Enumerable.Repeat(1, additional).ToArray();
        var sizes = Enumerable.Repeat(ArraySize, additional).ToArray();
        return Create(
            [..ones, ..Extents],
            [..sizes, ..Strides]);
    }

    public Shape Reduce(HashSet<Index> axis, bool keepDims = false)
    {
        var offsets = GetOffsets(axis);
        var extents = new List<int>(Dimensions);
        for (var i = 0; i < Dimensions; ++i)
        {
            if (!offsets.Contains(i))
            {
                extents.Add(Extents[i]);
            }
            else if (keepDims)
            {
                extents.Add(1);
            }
        }
        return Create(extents.ToArray());
    }
    
    public int this[int dimension] => Extents[dimension];

    public int this[Index dimension] => Extents[dimension];

    public Shape this[Range dimensions] => Create(Extents[dimensions]);
    
    public override string ToString() => $"({string.Join(",", Extents)})";

    public static unsafe int GetArraySize(ReadOnlySpan<int> extents)
    {
        var dims = extents.Length;
        var result = 1;

        fixed (int* sh = extents)
        {
            for (var i = dims - 1; i >= 0; --i)
            {
                result *= sh[i];
            }
        }

        return result;
    }
    
    public static unsafe int[] GetStrides(ReadOnlySpan<int> extents)
    {
        var dims = extents.Length;
        if (dims == 0)
            return [];

        var strides = new int[dims];

        fixed (int* sh = extents, st = strides)
        {
            st[dims - 1] = 1;

            for (var i = dims - 2; i >= 0; --i)
            {
                st[i] = st[i + 1] * sh[i + 1];
            }
        }

        return strides;
    }

    public int GetOffset(Index axis) => axis.GetOffset(Dimensions);
    
    public int[] GetOffsets(Index[] axis) =>
        axis
            .Select(GetOffset)
            .ToArray();

    public HashSet<int> GetOffsets(HashSet<Index> axis) =>
        axis
            .Select(GetOffset)
            .ToHashSet();

    IEnumerator<int> IEnumerable<int>.GetEnumerator() => 
        Extents
            .AsEnumerable()
            .GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => 
        Extents
            .AsEnumerable()
            .GetEnumerator();
}

