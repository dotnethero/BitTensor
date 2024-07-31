namespace BitTensor.CUDA.Abstractions;

public interface IDeviceArray : IDisposable
{
    void CopyToHost(Span<float> destination);
    void CopyToDevice(ReadOnlySpan<float> source);
}