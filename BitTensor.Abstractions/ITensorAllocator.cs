﻿namespace BitTensor.Abstractions;

public interface ITensorAllocator<out T> where T : ITensor<T>
{
    T Create(float value);
    T Create(float[] values);
    T Create(float[][] values);
    T Create(float[][][] values);
}
