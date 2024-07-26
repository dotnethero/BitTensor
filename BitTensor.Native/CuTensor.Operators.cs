using BitTensor.Abstractions;

namespace BitTensor.CUDA;

using static GenericOperations<CuTensor, CuBackend>;

public partial class CuTensor // Operators
{
    public static CuTensor operator +(CuTensor a, CuTensor b) => Add(a, b);
    public static CuTensor operator +(CuTensor a, float b) => Add(a, b);
    public static CuTensor operator +(float a, CuTensor b) => Add(a, b);
    public static CuTensor operator -(CuTensor a, CuTensor b) => Add(a, -b);
    public static CuTensor operator -(CuTensor a, float b) => Add(a, -b);
    public static CuTensor operator -(float a, CuTensor b) => Add(a, -b);
    public static CuTensor operator *(CuTensor a, CuTensor b) => Mul(a, b);
    public static CuTensor operator *(CuTensor a, float b) => Mul(a, b);
    public static CuTensor operator *(float a, CuTensor b) => Mul(a, b);
    public static CuTensor operator ^(CuTensor a, float b) => Pow(a, b);
    public static CuTensor operator -(CuTensor a) => Negate(a);
}
