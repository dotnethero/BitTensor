using System;
using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public class CudaVariable<T> : CudaNode<T> where T : unmanaged, IFloatingPoint<T>
{
    public override CudaContext Context { get; }
    public override CudaTensor<T> Tensor { get; }

    public CudaVariable(CudaContext context, CudaTensor<T> tensor) : base(tensor.Shape)
    {
        Context = context;
        Tensor = tensor;
    }

    public sealed override void EnsureHasUpdatedValues()
    {
        Outdated = false; // always up to date
    }

    public unsafe void LoadBatches(CudaDataset<T> dataset, ReadOnlySpan<int> batchIndexes)
    {
        var stride = dataset.Shape.Strides[0];

        for (var i = 0; i < batchIndexes.Length; i++)
        {
            var sampleIndex = batchIndexes[i];
            var source = dataset.Pointer + sampleIndex * stride;
            Tensor.Array.CopyToDevice(source, i * stride, stride);
        }

        Invalidate();
    }
}