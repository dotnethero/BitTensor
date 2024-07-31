using BitTensor.Core;
using NUnit.Framework;

namespace BitTensor.Tests.Basic;

[TestFixture]
class TensorIndexationTests
{
    [Test]
    public void Test_scalar()
    {
        var x = Tensor.Arrange(1, 9).Reshape([2, 2, 2]);

        Assert.That(x[0, 1, 0], Is.EqualTo(3));
        Assert.That(x[1, 0, 0], Is.EqualTo(5));
        Assert.That(x[1, 0, 1], Is.EqualTo(6));
        Assert.That(x[1, 1, 0], Is.EqualTo(7));
    }
}