﻿using BitTensor.Core;
using BitTensor.Core.Tests;
using NUnit.Framework;
using Python.Runtime;

namespace BitTensor.Tests.Comparison;

[TestFixture]
class SumComparisonTests
{
    [TestCase(new[] { 4 },       new int[] { })]
    [TestCase(new[] { 4 },       new[] { 0 })]
    [TestCase(new[] { 4, 3 },    new int[] { })]
    [TestCase(new[] { 4, 3 },    new[] { 0 })]
    [TestCase(new[] { 4, 3 },    new[] { 1 })]
    [TestCase(new[] { 4, 3 },    new[] { 0, 1 })]
    [TestCase(new[] { 4, 3, 7 }, new int[] { })]
    [TestCase(new[] { 4, 3, 7 }, new[] { 0 })]
    [TestCase(new[] { 4, 3, 7 }, new[] { 1 })]
    [TestCase(new[] { 4, 3, 7 }, new[] { 2 })]
    [TestCase(new[] { 4, 3, 7 }, new[] { 0, 1 })]
    [TestCase(new[] { 4, 3, 7 }, new[] { 1, 2 })]
    [TestCase(new[] { 4, 3, 7 }, new[] { 0, 2 })]
    [TestCase(new[] { 4, 3, 7 }, new[] { 0, 1, 2 })]
    public void Compare_sum_along_axis(int[] a, int[] ax)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL(); 
        
        var shape = $"[{string.Join(",", a)}]";
        var axis = $"[{string.Join(",", ax)}]";

        scope.ExecuteJax(
            $"""
             key = jax.random.PRNGKey(0)
             x = jax.random.normal(key, {shape})
             d = jnp.sum(x, axis={axis})
             """);

        var x = scope.GetTensor("x");
        var d = scope.GetTensor("d");

        var z = Tensor.Sum(x, axis: ax.ToHashSet());
        
        TensorAsserts.ShapesAreEqual(d, z);
        TensorAsserts.ValuesAreEqual(d, z);
    }
}