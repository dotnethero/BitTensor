using BitTensor.Abstractions;
using ILGPU;
using ILGPU.Runtime;

namespace BitTensor.CUDA;

using DType = float;
using DTypeView = ArrayView<float>;

internal readonly struct CuBackend : ITensorBackend<CuTensor>
{
    public static void ExecuteReshape(CuTensor a, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteBroadcast(CuTensor a, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteNegate(CuTensor a, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteSum(CuTensor a, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteSum(CuTensor a, HashSet<int> axes, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteAdd(CuTensor a, CuTensor b, CuTensor output)
    {
        var add = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DTypeView, DTypeView>(CuKernels.Add);
        add(output.Size, a.View, b.View, output.View);
    }

    public static void ExecuteAdd(CuTensor a, DType b, CuTensor output)
    {
        var add = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DType, DTypeView>(CuKernels.Add);
        add(output.Size, a.View, b, output.View);
    }

    public static void ExecuteMultiply(CuTensor a, CuTensor b, CuTensor output)
    {
        var mul = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DTypeView, DTypeView>(CuKernels.Mul);
        mul(output.Size, a.View, b.View, output.View);
    }

    public static void ExecuteMultiply(CuTensor a, DType b, CuTensor output)
    {
        var mul = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DType, DTypeView>(CuKernels.Mul);
        mul(output.Size, a.View, b, output.View);
    }
}
