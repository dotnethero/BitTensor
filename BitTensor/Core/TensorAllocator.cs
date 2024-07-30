using BitTensor.Abstractions;

namespace BitTensor.Core;

#pragma warning disable CS8500
public sealed class TensorAllocator : ITensorAllocator<Tensor>
{
    public Tensor Allocate(int[] shape) => Tensor.Allocate(shape);
    public Tensor Create(float value) => Tensor.Create(value);
    public Tensor Create(float[] values) => Tensor.Create(values);
    public Tensor Create(float[][] values) => Tensor.Create(values);
    public Tensor Create(float[][][] values) => Tensor.Create(values);
}