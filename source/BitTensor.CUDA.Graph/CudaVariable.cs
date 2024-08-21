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
    
    public void LoadBatches(Dataset<T> dataset, int[] batchIndexes)
    {
        var stride = dataset.Shape.Strides[0];

        for (var i = 0; i < batchIndexes.Length; i++)
        {
            var sampleIndex = batchIndexes[i];
            var batch = new ReadOnlySpan<T>(dataset.Data, sampleIndex * stride, stride);
            Tensor.Array.CopyToDevice(batch, i * stride, stride);
        }

        Invalidate();
    }
}