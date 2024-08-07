using BitTensor.Abstractions;
using BitTensor.CUDA;
using NUnit.Framework;

// ReSharper disable once CheckNamespace

namespace BitTensor.Core.Tests;

static class TensorAsserts
{
    public static void ShapesAreEqual(int[] expected, int[] actual) =>
        Assert.AreEqual(
            expected.Serialize(), 
            actual.Serialize());

    public static void ShapesAreEqual(CuTensor expected, CuTensor actual) =>
        ShapesAreEqual(expected.Shape, actual.Shape);

    public  static void ValuesAreEqual(CuTensor expected, CuTensor actual) =>
        CollectionAssert.AreEqual(
            expected.CopyToHost(), 
            actual.CopyToHost(),
            new ValuesComparer());

    public  static void ValuesAreEqual(CuTensor expected, CuTensor actual, float tolerance) =>
        CollectionAssert.AreEqual(
            expected.CopyToHost(), 
            actual.CopyToHost(),
            new ValuesComparer(tolerance));
}
