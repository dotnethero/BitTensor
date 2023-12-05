using BitTensor.Core;
using NUnit.Framework;

namespace BitTensor.Tests.Basic;

[TestFixture]
class TensorIndexationTests
{
    [Test]
    public void Test_slice()
    {
        var x = Tensor.Arrange(1, 9).Reshape([2, 2, 2]);

        Console.WriteLine(x.ToDataString());

        Console.WriteLine(x[0].ToDataString());
        Console.WriteLine(x[1].ToDataString());

        Console.WriteLine(x[0, 1].ToDataString());
        Console.WriteLine(x[1, 0].ToDataString());

        Assert.That(x[0].ToDataString(), Is.EqualTo(
            """
            [[   1.00   2.00 ]
             [   3.00   4.00 ]]
            """));
        
        Assert.That(x[1].ToDataString(), Is.EqualTo(
            """
            [[   5.00   6.00 ]
             [   7.00   8.00 ]]
            """));

        Assert.That(x[0, 1].ToDataString(), Is.EqualTo("[ 3.00 4.00 ]"));
        Assert.That(x[1, 0].ToDataString(), Is.EqualTo("[ 5.00 6.00 ]"));
    }

    [Test]
    public void Test_scalar()
    {
        var x = Tensor.Arrange(1, 9).Reshape([2, 2, 2]);

        Assert.That(x[0, 1, 0].Values.Scalar(), Is.EqualTo(3));
        Assert.That(x[1, 0, 0].Values.Scalar(), Is.EqualTo(5));
        Assert.That(x[1, 0, 1].Values.Scalar(), Is.EqualTo(6));
        Assert.That(x[1, 1, 0].Values.Scalar(), Is.EqualTo(7));
    }
}