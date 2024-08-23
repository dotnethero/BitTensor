using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;
using NUnit.Framework;

namespace BitTensor.Tests.Wrappers;

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
        using var descriptor = new CudnnTensorDescriptor<float>(Shape.Create(shape));
    }
}