namespace BitTensor.Abstractions;

public interface IDeviceArray
{
    float[] CopyToHost();
    void CopyToHost(Span<float> destination);
    void CopyToDevice(ReadOnlySpan<float> source);
}