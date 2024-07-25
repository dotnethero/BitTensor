using System.Runtime.CompilerServices;

namespace BitTensor.Core;

#pragma warning disable CS8500

internal interface IAllocation
{
    Span<float> Data { get; }
}

internal interface IAllocator
{
    IAllocation Allocate(int size);
}

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
}

internal class HostAllocator : IAllocator
{
    private static readonly Lazy<HostAllocator> Lazy = new(new HostAllocator());

    public static HostAllocator Instance => Lazy.Value;

    public IAllocation Allocate(int size) => new HostAllocation(size);
}