namespace BitTensor.CUDA.Abstractions;

public interface IDeviceArray : IDisposable
{
    unsafe void CopyToHost(Span<float> destination);
    unsafe void CopyToDevice(ReadOnlySpan<float> source);
}