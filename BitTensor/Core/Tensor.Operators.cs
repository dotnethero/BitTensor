using BitTensor.Abstractions;

namespace BitTensor.Core;

using static GenericOperations<Tensor, Backend>;

public partial class Tensor
{
    public static Tensor operator +(Tensor a, Tensor b) => Add(a, b);
    public static Tensor operator +(Tensor a, float b) => Add(a, b);
    public static Tensor operator +(float a, Tensor b) => Add(a, b);
    public static Tensor operator -(Tensor a, Tensor b) => Add(a, -b);
    public static Tensor operator -(Tensor a, float b) => Add(a, -b);
    public static Tensor operator -(float a, Tensor b) => Add(a, -b);
    public static Tensor operator *(Tensor a, Tensor b) => Mul(a, b);
    public static Tensor operator *(Tensor a, float b) => Mul(a, b);
    public static Tensor operator *(float a, Tensor b) => Mul(a, b);
    public static Tensor operator -(Tensor a) => Negate(a);
}
