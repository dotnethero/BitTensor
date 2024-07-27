using BitTensor.Abstractions;
using ILGPU;
using ILGPU.Runtime;

namespace BitTensor.CUDA;

using DType = float;
using DTypeView = ArrayView<float>;

internal class CuAllocator(Accelerator accelerator) : ITensorAllocator<CuTensor>
{
    public CuTensor AllocateOnes(int[] shape)
    {
        var tensor = new CuTensor(accelerator, shape);
        var add = accelerator.LoadAutoGroupedStreamKernel<Index1D, DType, DTypeView>(CuKernels.Memset);
        add(tensor.Size, 1, tensor.Buffer.View);
        return tensor;
    }

    public CuTensor AllocateZeros(int[] shape)
    {
        return new CuTensor(accelerator, shape);
    }
}
