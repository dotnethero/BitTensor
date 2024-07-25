using System.Runtime.CompilerServices;

namespace BitTensor.Core;

#pragma warning disable CS8500

public interface IAllocation
{
    Span<float> Data { get; }
    void CopyToDevice(float[] source);
    void CopyToHost(float[] destination);
}

public interface IAllocator
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

    public void CopyToDevice(float[] values)
    {
    }

    public void CopyToHost(float[] destination)
    {
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