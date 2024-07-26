using BitTensor.Abstractions;

namespace BitTensor.CUDA;

public readonly struct CuBackend : ITensorBackend<CuTensor>
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
        throw new NotImplementedException();
    }

    public static void ExecuteAdd(CuTensor a, float b, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteMultiply(CuTensor a, CuTensor b, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecuteMultiply(CuTensor a, float b, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void ExecutePower(CuTensor a, float b, CuTensor output)
    {
        throw new NotImplementedException();
    }
}
