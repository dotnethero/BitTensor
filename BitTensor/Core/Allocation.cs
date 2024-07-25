using System.Runtime.CompilerServices;

namespace BitTensor.Core;

#pragma warning disable CS8500

internal interface IAllocation
{
    Span<float> Data { get; }
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