namespace BitTensor.Abstractions;

public interface IDeviceArray<T> where T : unmanaged
{
    int ElementSize { get; }
    int Size { get; }
    unsafe T* Pointer { get; }

    void CopyToDevice(ReadOnlySpan<T> source);

    void CopyToHost(Span<T> destination);
    
    T[] CopyToHost()
    {
        var destination = new T[Size];
        CopyToHost(destination);
        return destination;
    }
}

