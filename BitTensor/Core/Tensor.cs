#pragma warning disable CS8500 // This takes the address of, gets the size of, or declares a pointer to a managed type

using System.Runtime.CompilerServices;
using BitTensor.Abstractions;

namespace BitTensor.Core;

public sealed class TensorAllocator : ITensorAllocator<Tensor>
{
    public Tensor Allocate(int[] shape) => Tensor.Allocate(shape);
    public Tensor Create(float value) => Tensor.Create(value);
    public Tensor Create(float[] values) => Tensor.Create(values);
    public Tensor Create(float[][] values) => Tensor.Create(values);
    public Tensor Create(float[][][] values) => Tensor.Create(values);
}

public sealed partial class Tensor : AbstractTensorNode<Tensor>, ITensorNode<Tensor>, IMutableTensor<Tensor>, IHasAllocator<Tensor>
{
    internal float[] Data;
    internal Lazy<Tensor> TransposeLazy;

    public ReadOnlySpan<float> Values
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            EnsureHasUpdatedValues();
            return Data;
        }
    }

    public ITensorAllocator<Tensor> Allocator { get; } = new TensorAllocator();

    public Tensor T
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => TransposeLazy.Value;
    }

    // utility
    public int BatchDimension = 0;

    internal Tensor(int[] shape, float[]? values = null) : base(shape)
    {
        Data = values ?? new float[Size];
        TransposeLazy = new(Transpose);
    }

    internal Tensor(int[] shape, Tensor[] children, ForwardFunction forward, BackwardFunction backward, float[]? values = null) : base(shape, children, forward, backward)
    {
        Data = values ?? new float[Size];
        TransposeLazy = new(Transpose);
    }
    
    // Reshape
    internal Tensor(int[] shape, Tensor tensor) : base(shape, [tensor], _ => {}, (grad, self) => [CreateReshape(tensor.Shape, grad)])
    {
        Data = tensor.Data;
        TransposeLazy = new(Transpose);
    }

    public static Tensor CreateNode(int[] shape, Tensor[] children, ForwardFunction forward, BackwardFunction backward)
    {
        return new Tensor(shape, children, forward, backward);
    }

    public static Tensor CreateReshape(int[] shape, Tensor source)
    {
        return new Tensor(shape, source);
    }

    public void ApplyOffset(Tensor offset)
    {
        Backend.ExecuteAdd(this, offset, this);
    }

    public void ApplyScale(Tensor scale)
    {
        Backend.ExecuteMultiply(this, scale, this);
    }
}
