﻿using BitTensor.Abstractions;
using BitTensor.CUDA.Graph;
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
    
    public static void ShapesAreEqual(AbstractTensor expected, CuTensorNode actual) =>
        ShapesAreEqual(expected.Shape, actual.Shape);

    public static void ValuesAreEqual(IDeviceArray expected, IDeviceArray actual) =>
        CollectionAssert.AreEqual(
            expected.CopyToHost(), 
            actual.CopyToHost(),
            new ValuesComparer());

    public static void ValuesAreEqual(IDeviceArray expected, IDeviceArray actual, float tolerance) =>
        CollectionAssert.AreEqual(
            expected.CopyToHost(), 
            actual.CopyToHost(),
            new ValuesComparer(tolerance));

    public static void ValuesAreEqual(IDeviceArray expected, CuTensorNode actual)
    {
        actual.EnsureHasUpdatedValues();
        ValuesAreEqual(expected, actual.Tensor);
    }
}
