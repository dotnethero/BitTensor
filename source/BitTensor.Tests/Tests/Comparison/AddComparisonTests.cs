using BitTensor.Core.Tests;
using NUnit.Framework;
using Python.Runtime;

namespace BitTensor.Tests.Comparison;

[TestFixture]
class AddComparisonTests
{
    [Test]
    [TestCase(new[] { 1 }, new[] { 1 })]
    [TestCase(new[] { 1 }, new[] { 7 })]
    [TestCase(new[] { 1 }, new[] { 1, 7 })]
    [TestCase(new[] { 1 }, new[] { 6, 7 })]
    [TestCase(new[] { 1 }, new[] { 6, 7, 1 })]
    [TestCase(new[] { 5 }, new[] { 6, 1 })]
    [TestCase(new[] { 5 }, new[] { 6, 5 })]
    [TestCase(new[] { 5 }, new[] { 6, 7, 1 })]
    [TestCase(new[] { 3, 1 }, new[] { 1 })]
    [TestCase(new[] { 3, 1 }, new[] { 7 })]
    [TestCase(new[] { 3, 1 }, new[] { 1, 7 })]
    [TestCase(new[] { 3, 1 }, new[] { 3, 7 })]
    [TestCase(new[] { 3, 1 }, new[] { 6, 3, 7 })]
    [TestCase(new[] { 3, 5 }, new[] { 3, 1 })]
    [TestCase(new[] { 3, 5 }, new[] { 3, 5 })]
    [TestCase(new[] { 3, 5 }, new[] { 1, 5 })]
    [TestCase(new[] { 3, 5 }, new[] { 6, 3, 5 })]
    [TestCase(new[] { 3, 5 }, new[] { 6, 3, 1 })]
    [TestCase(new[] { 3, 5 }, new[] { 6, 1, 5 })]
    public void Compare_add_different_shapes(int[] a, int[] b)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL(); 
        
        var x_py_shape = $"[{string.Join(",", a)}]";
        var y_py_shape = $"[{string.Join(",", b)}]";

        scope.ExecuteJax(
            $"""
             key = jax.random.PRNGKey(0)
             xk, yk = jax.random.split(key)
             x = jax.random.normal(xk, {x_py_shape})
             y = jax.random.normal(yk, {y_py_shape})
             d = jnp.add(x, y)
             """);

        var x = scope.GetTensor("x");
        var y = scope.GetTensor("y");
        var d = scope.GetTensor("d");

        var z1 = x + y;
        var z2 = y + x;

        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(d, z1);
            TensorAsserts.ShapesAreEqual(d, z2);
        });
        
        Assert.Multiple(() =>
        {
            TensorAsserts.ValuesAreEqual(d, z1);
            TensorAsserts.ValuesAreEqual(d, z2);
        });
    }
}