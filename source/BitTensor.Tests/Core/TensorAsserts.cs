using System.Diagnostics;
using BitTensor.Abstractions;
using NUnit.Framework;

// ReSharper disable once CheckNamespace

namespace BitTensor.Core.Tests;

static class TensorAsserts
{
    [StackTraceHidden]
    public static void ShapesAreEqual(Shape expected, Shape actual) =>
        Assert.AreEqual(
            expected.ToString(), 
            actual.ToString());

    [StackTraceHidden]
    public static void ShapesAreEqual(AbstractTensor expected, AbstractTensor actual) =>
        ShapesAreEqual(expected.Shape, actual.Shape);
    
    [StackTraceHidden]
    public static void ValuesAreEqual(IDeviceArray<float> expected, IDeviceArray<float> actual, float tolerance = 3e-5f) =>
        CollectionAssert.AreEqual(
            expected.CopyToHost(), 
            actual.CopyToHost(),
            new ValuesComparer(tolerance));
}
