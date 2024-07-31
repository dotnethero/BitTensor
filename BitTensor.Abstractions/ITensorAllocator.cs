namespace BitTensor.Abstractions;

public interface ITensorAllocator<out T> where T : ITensor<T>
{
    T Allocate(int[] shape);
    T Create(float value);
    T Create(float[] values);
    T Create(float[][] values);
    T Create(float[][][] values);
}

public interface IHasAllocator<out T> where T : ITensor<T>
{
    ITensorAllocator<T> Allocator { get; }
}
