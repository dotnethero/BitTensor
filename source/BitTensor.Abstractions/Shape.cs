using System.Collections;
using System.Runtime.CompilerServices;

namespace BitTensor.Abstractions;

[CollectionBuilder(typeof(Shape), "Create")]
public sealed class Shape : IEnumerable<int>
{
    public readonly int Dimensions;
    public readonly int ArraySize;
    public readonly int[] Extents;
    public readonly int[] Strides;

    public static Shape Create(ReadOnlySpan<int> extents) => new(extents);

    private Shape(ReadOnlySpan<int> extents)
    {
        Extents = extents.ToArray();
        Dimensions = extents.Length;
        ArraySize = GetArraySize(extents);
        Strides = GetStrides(extents);
    }

    public Shape Transpose(int[] axis)
    {
        var extents = new int[Dimensions];
        for (var i = 0; i < Dimensions; ++i)
        {
            extents[i] = Extents[axis[i]];
        }
        return Create(extents);
    }

    public Shape Append(int size) => Create([..Extents, size]);

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
    
    public HashSet<int> GetOffsets(HashSet<Index> axis) =>
        axis
            .Select(x => x.GetOffset(Dimensions))
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

