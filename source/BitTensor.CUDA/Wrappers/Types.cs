using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using static cuTENSOR;

internal static unsafe class Types
{
    public static cutensorDataType_t GetDataType<T>()
        where T : INumberBase<T> =>
        T.Zero switch
        {
            Half => cutensorDataType_t.CUTENSOR_R_16F,
            float => cutensorDataType_t.CUTENSOR_R_32F,
            double => cutensorDataType_t.CUTENSOR_R_64F,
            _ => throw new NotSupportedException($"Unsupported element type: {typeof(T).FullName}")
        };
    
    public static cutensorComputeDescriptor* GetComputeType<T>()
        where T : INumberBase<T> =>
        T.Zero switch
        {
            Half => CUTENSOR_COMPUTE_DESC_32F,
            float => CUTENSOR_COMPUTE_DESC_32F,
            double => CUTENSOR_COMPUTE_DESC_64F,
            _ => throw new NotSupportedException($"Unsupported operation element type: {typeof(T).FullName}")
        };
}
