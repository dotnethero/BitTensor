using BitTensor.Abstractions;
using ILGPU;
using ILGPU.Runtime;

namespace BitTensor.CUDA;

using DType = float;
using DTypeView = ArrayView<float>;

public readonly unsafe struct CuBackend : ITensorBackend<CuTensor>
{
    private static void Add(Index1D i, DTypeView a, DTypeView b, DTypeView output)
    {
        output[i] = a[i] + b[i];
    }

    private static void Add(Index1D i, DTypeView a, DType b, DTypeView output)
    {
        output[i] = a[i] + b;
    }

    private static void Mul(Index1D i, DTypeView a, DTypeView b, DTypeView output)
    {
        output[i] = a[i] * b[i];
    }
    
    private static void Mul(Index1D i, DTypeView a, DType b, DTypeView output)
    {
        output[i] = a[i] * b;
    }

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
        var add = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DTypeView, DTypeView>(Add);
        add(output.Size, a.Buffer.View, b.Buffer.View, output.Buffer.View);
    }

    public static void ExecuteAdd(CuTensor a, float b, CuTensor output)
    {
        var add = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DType, DTypeView>(Add);
        add(output.Size, a.Buffer.View, b, output.Buffer.View);
    }

    public static void ExecuteMultiply(CuTensor a, CuTensor b, CuTensor output)
    {
        var mul = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DTypeView, DTypeView>(Mul);
        mul(output.Size, a.Buffer.View, b.Buffer.View, output.Buffer.View);
    }

    public static void ExecuteMultiply(CuTensor a, float b, CuTensor output)
    {
        var mul = output.Accelerator.LoadAutoGroupedStreamKernel<Index1D, DTypeView, DType, DTypeView>(Mul);
        mul(output.Size, a.Buffer.View, b, output.Buffer.View);
    }

    public static void ExecutePower(CuTensor a, float b, CuTensor output)
    {
        throw new NotImplementedException();
    }
}
