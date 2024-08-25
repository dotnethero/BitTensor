namespace BitTensor.Abstractions;

public interface IDeviceArray : IDisposable
{
    int ElementSize { get; }
    int Size { get; }
}

public interface IDeviceArray<T> : IDeviceArray where T : unmanaged
{
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

public interface IUniqueDeviceArray<T> : IDeviceArray<T> where T : unmanaged
{
    long UniqueId { get; }
}
