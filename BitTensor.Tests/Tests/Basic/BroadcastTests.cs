using BitTensor.Abstractions;
using NUnit.Framework;

namespace BitTensor.Tests.Basic;

[TestFixture]
public class TensorShapeTests
{
    [Test]
    public void GetBroadcastedAxis_SameShape_ReturnsEmptyArray()
    {
        int[] inputShape  = [2, 3, 4];
        int[] resultShape = [2, 3, 4];
        int[] result = Shapes.GetBroadcastedAxis(inputShape, resultShape);

        Assert.IsEmpty(result);
    }

    [Test]
    public void GetBroadcastedAxis_ScalarInput_ReturnsAllAxes()
    {
        int[] inputShape = Array.Empty<int>();
        int[] resultShape = [2, 3, 4];
        int[] result = Shapes.GetBroadcastedAxis(inputShape, resultShape);

        CollectionAssert.AreEquivalent(new[] { 0, 1, 2 }, result);
    }

    [Test]
    public void GetBroadcastedAxis_BroadcastLastDimension_ReturnsLastAxis()
    {
        int[] inputShape  = [2, 3, 1];
        int[] resultShape = [2, 3, 4];
        int[] result = Shapes.GetBroadcastedAxis(inputShape, resultShape);

        CollectionAssert.AreEquivalent(new[] { 2 }, result);
    }

    [Test]
    public void GetBroadcastedAxis_BroadcastMiddleDimension_ReturnsMiddleAxis()
    {
        int[] inputShape  = [2, 1, 4];
        int[] resultShape = [2, 3, 4];
        int[] result = Shapes.GetBroadcastedAxis(inputShape, resultShape);

        CollectionAssert.AreEquivalent(new[] { 1 }, result);
    }

    [Test]
    public void GetBroadcastedAxis_BroadcastMultipleDimensions_ReturnsMultipleAxes()
    {
        int[] inputShape  = [1, 3, 1];
        int[] resultShape = [2, 3, 4];
        int[] result = Shapes.GetBroadcastedAxis(inputShape, resultShape);

        CollectionAssert.AreEquivalent(new[] { 0, 2 }, result);
    }

    [Test]
    public void GetBroadcastedAxis_InputHigherRank_ReturnsEmptyArray()
    {
        int[] inputShape  = [2, 3, 4, 5];
        int[] resultShape =    [3, 4, 5];
        int[] result = Shapes.GetBroadcastedAxis(inputShape, resultShape);

        Assert.IsEmpty(result);
    }

    [Test]
    public void GetBroadcastedAxis_InputLowerRank_ReturnsPrependedAxes()
    {
        int[] inputShape  =       [4, 5];
        int[] resultShape = [2, 3, 4, 5];
        int[] result = Shapes.GetBroadcastedAxis(inputShape, resultShape);

        CollectionAssert.AreEquivalent(new[] { 0, 1 }, result);
    }

    [Test]
    public void GetBroadcastedAxis_MixedBroadcasting_ReturnsCorrectAxes()
    {
        int[] inputShape  =    [1, 5, 1];
        int[] resultShape = [2, 5, 5, 4];
        int[] result = Shapes.GetBroadcastedAxis(inputShape, resultShape);

        CollectionAssert.AreEquivalent(new[] { 0, 1, 3 }, result);
    }

    [TestCase(new int[] { },   new int[] { },   new int[] { })]
    [TestCase(new int[] { 1 }, new int[] { 5 }, new int[] { 0 })]
    [TestCase(new int[] { 5 }, new int[] { 5 }, new int[] { })]
    public void GetBroadcastedAxis_EdgeCases_HandledCorrectly(int[] inputShape, int[] resultShape, int[] expected)
    {
        int[] result = Shapes.GetBroadcastedAxis(inputShape, resultShape);

        CollectionAssert.AreEquivalent(expected, result);
    }
}