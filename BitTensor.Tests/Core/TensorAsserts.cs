using NUnit.Framework;

// ReSharper disable once CheckNamespace

namespace BitTensor.Core.Tests;

static class TensorAsserts
{
    public static void ShapesAreEqual(int[] expected, int[] actual) =>
        Assert.AreEqual(
            expected.Serialize(), 
            actual.Serialize());

    public static void ShapesAreEqual(Tensor expected, Tensor actual) =>
        ShapesAreEqual(expected.Shape, actual.Shape);

    public  static void ValuesAreEqual(Tensor expected, Tensor actual) =>
        CollectionAssert.AreEqual(
            expected.Values.ToArray(), 
            actual.Values.ToArray(),
            new ValuesComparer());
}
