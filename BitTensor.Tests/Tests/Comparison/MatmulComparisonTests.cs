using BitTensor.Core;
using BitTensor.Core.Tests;
using NUnit.Framework;
using Python.Runtime;

namespace BitTensor.Tests.Comparison;

[TestFixture]
class MatmulComparisonTests
{
    [Test]
    [TestCase(1, 1, 1)]
    [TestCase(1, 3, 1)]
    [TestCase(1, 1, 12)]
    [TestCase(1, 3, 12)]
    [TestCase(4, 1, 1)]
    [TestCase(4, 3, 1)]
    [TestCase(4, 1, 12)]
    [TestCase(4, 3, 12)]
    public void Compare_2Dx2D_matmul(int a, int b, int c)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL();

        scope.ExecuteJax(
            $"""
            key = jax.random.PRNGKey(0)
            x = jax.random.normal(key, [{a}, {b}])
            y = jax.random.normal(key, [{b}, {c}])
            z = jnp.dot(x, y)
            """);

        var x = scope.Get2D("x");;
        var y = scope.Get2D("y");
        var z_true = scope.Get2D("z");
        var z = Tensor.Matmul(x, y);

        TensorAsserts.ShapesAreEqual(z_true, z);
        TensorAsserts.ValuesAreEqual(z_true, z);
    }

    [Test]
    [TestCase(1, 1)]
    [TestCase(3, 1)]
    [TestCase(1, 12)]
    [TestCase(3, 12)]
    public void Compare_2Dx1Dx2D_matmul(int n, int m)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL();

        scope.ExecuteJax(
            $"""
             key = jax.random.PRNGKey(0)
             a = jax.random.normal(key, [{n}])
             b = jax.random.normal(key, [{n}, {m}])
             c = jax.random.normal(key, [{m}, {n}])
             ab = jnp.dot(a, b)
             ca = jnp.dot(c, a)
             """);

        var ab_shape = scope.Eval<int[]>("ab.shape")!;
        var ca_shape = scope.Eval<int[]>("ca.shape")!;

        Console.WriteLine(ab_shape.Serialize());
        Console.WriteLine(ca_shape.Serialize());

        var a = scope.Get1D("a");
        var b = scope.Get2D("b");
        var c = scope.Get2D("c");
        var ab_true = scope.Get1D("ab");
        var ca_true = scope.Get1D("ca");

        var ab = Tensor.Matmul(a, b);
        var ca = Tensor.Matmul(c, a);

        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(ab_true, ab);
            TensorAsserts.ShapesAreEqual(ca_true, ca);
            TensorAsserts.ValuesAreEqual(ab_true, ab);
            TensorAsserts.ValuesAreEqual(ca_true, ca);
        });
    }

    [Test]
    [TestCase(1, 1)]
    [TestCase(3, 1)]
    [TestCase(1, 12)]
    [TestCase(3, 12)]
    public void Compare_1Dx2Dx1D_matmul(int n, int m)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL();

        scope.ExecuteJax(
            $"""
             key = jax.random.PRNGKey(0)
             a = jax.random.normal(key, [{n}])
             b = jax.random.normal(key, [{m}])
             c = jax.random.normal(key, [{n}, {m}])
             ac = jnp.dot(a, c)
             cb = jnp.dot(c, b)
             """);

        var ac_shape = scope.Eval<int[]>("ac.shape")!;
        var cb_shape = scope.Eval<int[]>("cb.shape")!;

        Console.WriteLine(ac_shape.Serialize());
        Console.WriteLine(cb_shape.Serialize());

        var a = scope.Get1D("a");
        var b = scope.Get1D("b");
        var c = scope.Get2D("c");
        var ac_true = scope.Get1D("ac");
        var cb_true = scope.Get1D("cb");

        var ac = Tensor.Matmul(a, c);
        var cb = Tensor.Matmul(c, b);

        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(ac_true, ac);
            TensorAsserts.ShapesAreEqual(cb_true, cb);
            TensorAsserts.ValuesAreEqual(ac_true, ac);
            TensorAsserts.ValuesAreEqual(cb_true, cb);
        });
    }
}
