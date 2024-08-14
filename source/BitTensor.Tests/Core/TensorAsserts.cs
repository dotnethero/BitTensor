﻿using BitTensor.Abstractions;
using NUnit.Framework;

// ReSharper disable once CheckNamespace

namespace BitTensor.Core.Tests;

static class TensorAsserts
{
    public static void ShapesAreEqual(Shape expected, Shape actual) =>
        Assert.AreEqual(
            expected.ToString(), 
            actual.ToString());

    public static void ShapesAreEqual(AbstractTensor expected, AbstractTensor actual) =>
        ShapesAreEqual(expected.Shape, actual.Shape);
    
    public static void ValuesAreEqual(IDeviceArray<float> expected, IDeviceArray<float> actual, float tolerance = 1e-5f) =>
        CollectionAssert.AreEqual(
            expected.CopyToHost(), 
            actual.CopyToHost(),
            new ValuesComparer(tolerance));
}
