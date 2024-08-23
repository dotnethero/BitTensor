using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal static class Types
{
    public static cudnnDataType_t GetDataType<T>()
        where T : IFloatingPoint<T> =>
        T.Zero switch
        {
            Half => cudnnDataType_t.CUDNN_DATA_HALF,
            float => cudnnDataType_t.CUDNN_DATA_FLOAT,
            double => cudnnDataType_t.CUDNN_DATA_DOUBLE,
            _ => throw new NotSupportedException($"Unsupported element type: {typeof(T).FullName}")
        };
}
