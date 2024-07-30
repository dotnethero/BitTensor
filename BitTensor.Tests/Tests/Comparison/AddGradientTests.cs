using BitTensor.Abstractions;
using BitTensor.Core;
using BitTensor.Core.Tests;
using NUnit.Framework;
using Python.Runtime;

namespace BitTensor.Tests.Comparison;

[TestFixture]
class AddGradientTests
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
    public void Compare_add_operation_gradients(int[] a, int[] b)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL(); 
        
        var x_py_shape = $"[{string.Join(",", a)}]";
        var y_py_shape = $"[{string.Join(",", b)}]";

        scope.ExecuteJax(
            $"""
             def func(A, B):
                return jnp.sum(jnp.add(A, B))

             key = jax.random.PRNGKey(0)
             xk, yk = jax.random.split(key)
             x = jax.random.normal(xk, {x_py_shape})
             y = jax.random.normal(yk, {y_py_shape})

             xy_dx = jax.grad(func, argnums=0)(x, y)
             xy_dy = jax.grad(func, argnums=1)(x, y)
             """);

        var x = scope.GetTensor("x");
        var y = scope.GetTensor("y");
        
        var xy_dx_true = scope.GetTensor("xy_dx");
        var xy_dy_true = scope.GetTensor("xy_dy");
        
        var xy_grads = Auto.GetGradients(Tensor.Sum(x + y));
        var xy_dx = xy_grads[x];
        var xy_dy = xy_grads[y];

        var yx_grads = Auto.GetGradients(Tensor.Sum(y + x));
        var yx_dx = yx_grads[x];
        var yx_dy = yx_grads[y];

        
        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(xy_dx_true, xy_dx);
            TensorAsserts.ShapesAreEqual(xy_dy_true, xy_dy);
            TensorAsserts.ShapesAreEqual(xy_dx_true, yx_dx);
            TensorAsserts.ShapesAreEqual(xy_dy_true, yx_dy);
        });

        Assert.Multiple(() => 
        {
            TensorAsserts.ValuesAreEqual(xy_dx_true, xy_dx);
            TensorAsserts.ValuesAreEqual(xy_dy_true, xy_dy);
            TensorAsserts.ValuesAreEqual(xy_dx_true, yx_dx);
            TensorAsserts.ValuesAreEqual(xy_dy_true, yx_dy);
        });
    }
}