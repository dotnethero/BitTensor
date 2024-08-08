namespace BitTensor.CUDA.Graph;

public static class CuTensorExtensions
{
    public static CuTensorNode ToNode(this CuTensor tensor) => new(tensor, owned: true);
}