using System.Runtime.CompilerServices;

namespace BitTensor.Abstractions;

public interface IAllocation
{
    void CopyToDevice(ReadOnlySpan<float> source);
    void CopyToHost(Span<float> destination);

    Span<float> Data
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get;
    }
}

public interface IAllocator
{
    IAllocation Allocate(int size);
}
