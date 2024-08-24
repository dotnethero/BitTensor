using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;
using NUnit.Framework;

namespace BitTensor.Tests.cuDNN;

[TestFixture]
class WrapperTests
{
    [Test]
    [TestCase(new int[] { })]
    [TestCase(new[] { 2 })]
    [TestCase(new[] { 2, 3 })]
    [TestCase(new[] { 2, 3, 4 })]
    [TestCase(new[] { 2, 3, 4, 5 })]
    [TestCase(new[] { 2, 3, 4, 5, 6 })]
    public void Create_cuDNN_tensor_descriptor(int[] shape)
    {
        using var descriptor = new CudnnTensorDescriptor<float>(Shape.Create(shape), 1);
    }

    [TestCase(new[] { 4, 4 }, new[] { 4, 4 }, new[] { 4, 4 })]
    [TestCase(new[] { 2, 3 }, new[] { 3, 4 }, new[] { 2, 4 })]
    public void Create_cuDNN_matmul_descriptor(int[] aShape, int[] bShape, int[] cShape)
    {
        using var a = CreateTensorDescriptor(1, aShape);
        using var b = CreateTensorDescriptor(2, bShape);
        using var c = CreateTensorDescriptor(3, cShape);
        
        using var mmc = new CudnnMatMulOperator<float>();
        using var mm = new CudnnMatMulOperation<float>(mmc, a, b, c);
    }
    
    [TestCase(new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 })]
    [TestCase(new[] { 2, 2 }, new[] { 2, 2 }, new[] { 2, 2 })]
    public void Create_cuDNN_pointwise_descriptor(int[] aShape, int[] bShape, int[] cShape)
    {
        using var a = CreateTensorDescriptor(1, aShape);
        using var b = CreateTensorDescriptor(2, bShape);
        using var c = CreateTensorDescriptor(3, cShape);

        using var pwc = new CudnnPointwiseOperator<float>();
        using var pw = new CudnnPointwiseOperation<float>(pwc, a, b, c);
    }

    private static CudnnTensorDescriptor<float> CreateTensorDescriptor(long id, int[] shape) => new(Shape.Create(shape), id);
}