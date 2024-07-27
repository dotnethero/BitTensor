namespace BitTensor.Abstractions;

public interface ITensorAllocator<out T> where T : ITensor<T>
{
    T AllocateOnes(int[] shape);
    T AllocateOne() => AllocateOnes([]);
    T AllocateZeros(int[] shape);
    T AllocateZero() => AllocateZeros([]);
}
