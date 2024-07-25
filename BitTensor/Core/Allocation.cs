using System.Runtime.CompilerServices;
using BitTensor.Abstractions;

namespace BitTensor.Core;

#pragma warning disable CS8500

internal readonly struct HostAllocation(float[] data) : IAllocation
{
    public Span<float> Data
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => data;
    }

    public HostAllocation(int size) : this(new float[size])
    {
    }

    public void CopyToDevice(ReadOnlySpan<float> source)
    {
    }

    public void CopyToHost(Span<float> destination)
    {
    }
}

internal class HostAllocator : IAllocator
{
    private static readonly Lazy<HostAllocator> Lazy = new(new HostAllocator());

    public static HostAllocator Instance => Lazy.Value;

    public IAllocation Allocate(int size) => new HostAllocation(size);
}