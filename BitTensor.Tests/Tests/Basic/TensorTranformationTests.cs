using BitTensor.Core;
using BitTensor.Core.Tests;
using NUnit.Framework;

namespace BitTensor.Tests.Basic;

[TestFixture]
class TensorTranformationTests
{
    [Test]
    public static void Test_reshape()
    {
        var a = Tensor.Create(10);

        Console.WriteLine(a.ToDataString());

        var x = Tensor.Arrange(1, 19);
        var y = x.Reshape([2, 3, 3]);

        Console.WriteLine(y.ToDataString());

        Assert.That(y.Shape, Is.EqualTo(new[] {2, 3, 3}));
        Assert.That(y[0, 2, 2], Is.EqualTo(9));
        Assert.That(y[1, 0, 2], Is.EqualTo(12));
        Assert.That(y[1, 1, 0], Is.EqualTo(13));
    }

    [Test]
    public void Test_transpose_pseudo_3D_102()
    {
        var x = Tensor.Arrange(1, 17).Reshape([4, 4, 1]);
        var t = x.Transpose([1, 0, 2]);

        Console.WriteLine(x.ToDataString(dimsPerLine: 2));
        Console.WriteLine(t.ToDataString(dimsPerLine: 2));
        
        Assert.That(t.Shape, Is.EqualTo(new[] {4, 4, 1}));
        Assert.That(t.ToDataString(2), Is.EqualTo(
            """
            [[[   1.00 ]  [   5.00 ]  [   9.00 ]  [  13.00 ]]
             [[   2.00 ]  [   6.00 ]  [  10.00 ]  [  14.00 ]]
             [[   3.00 ]  [   7.00 ]  [  11.00 ]  [  15.00 ]]
             [[   4.00 ]  [   8.00 ]  [  12.00 ]  [  16.00 ]]]
            """));
    }
    
    [Test]
    public void Test_transpose_pseudo_3D_021()
    {
        var x = Tensor.Arrange(1, 17).Reshape([1, 4, 4]);
        var t = x.Transpose([0, 2, 1]);

        Console.WriteLine(x.ToDataString());
        Console.WriteLine(t.ToDataString());

        Assert.That(t.Shape, Is.EqualTo(new[] {1, 4, 4}));
        Assert.That(t.ToDataString(), Is.EqualTo(
            """
            [[[   1.00   5.00   9.00  13.00 ]
              [   2.00   6.00  10.00  14.00 ]
              [   3.00   7.00  11.00  15.00 ]
              [   4.00   8.00  12.00  16.00 ]]]
            """));
    }
    
    [Test]
    public void Test_transpose_pseudo_3D_021_100_000()
    {
        var x = Tensor.Arrange(1, 1001).Reshape([10, 10, 10]);

        for (var i = 0; i < 10_000; i++)
        {
            x.Transpose();
        }
    }

    [Test]
    public void Test_transpose_3D_201()
    {
        var x = Tensor.Arrange(1, 9).Reshape([2, 2, 2]);
        var t = x.Transpose([2, 0, 1]);

        Console.WriteLine(x.ToDataString());
        Console.WriteLine(t.ToDataString());

        Assert.That(t.Shape, Is.EqualTo(new[] {2, 2, 2}));
        Assert.That(t.ToDataString(), Is.EqualTo(
            """
            [[[   1.00   3.00 ]
              [   5.00   7.00 ]]
             [[   2.00   4.00 ]
              [   6.00   8.00 ]]]
            """));
    }
    
    [Test]
    public void Test_transpose_3D_021()
    {
        var x = Tensor.Arrange(1, 9).Reshape([2, 2, 2]);
        var t = x.Transpose([0, 2, 1]);

        Console.WriteLine(x.ToDataString());
        Console.WriteLine(t.ToDataString());

        Assert.That(t.Shape, Is.EqualTo(new[] {2, 2, 2}));
        Assert.That(t.ToDataString(), Is.EqualTo(
            """
            [[[   1.00   3.00 ]
              [   2.00   4.00 ]]
             [[   5.00   7.00 ]
              [   6.00   8.00 ]]]
            """));
    }

    [Test]
    public void Test_transpose_identity()
    {
        var x = Tensor.Arrange(1, 10).Reshape([3, 3]);
        var t = x.Transpose([0, 1]); // No change in axes

        Console.WriteLine(t.ToDataString());

        TensorAsserts.ShapesAreEqual([3, 3], t.Shape);
        TensorAsserts.ValuesAreEqual(x, t);

        Assert.That(t.ToDataString(), Is.EqualTo(
            """
            [[   1.00   2.00   3.00 ]
             [   4.00   5.00   6.00 ]
             [   7.00   8.00   9.00 ]]
            """));
    }
}