using BitTensor.Abstractions;
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

    [Test]
    // Simple cases
    [TestCase(new[] { 1 }, new[] { 1 })] // Scalar multiplication
    [TestCase(new[] { 2, 3 }, new[] { 3, 4 })] // Regular matrix multiplication

    // Vector and matrix multiplication
    [TestCase(new[] { 3 }, new[] { 3, 2 })] // Vector and matrix
    [TestCase(new[] { 2, 3 }, new[] { 3 })] // Matrix and vector

    // Broadcasting cases
    [TestCase(new[] { 2, 1 }, new[] { 1, 2 })] // Broadcasting with different dimensions
    [TestCase(new[] { 1, 3 }, new[] { 3, 2 })] // Broadcasting with a vector
    [TestCase(new[] { 4, 1, 3 }, new[] { 3, 5 })] // Broadcasting in batched dimensions
    [TestCase(new[] { 1, 4, 3 }, new[] { 3, 2 })] // Broadcasting in batched dimensions

    // Batched multiplication cases
    [TestCase(new[] { 5, 2, 3 }, new[] { 3, 4 })] // Batched matrix multiplication
    [TestCase(new[] { 5, 2, 3 }, new[] { 5, 3, 4 })] // Batched matrix multiplication with same batch size
    [TestCase(new[] { 3, 1, 2, 3 }, new[] { 3, 3, 4 })] // Higher-dimensional batched multiplication
    [TestCase(new[] { 2, 3, 1, 3 }, new[] { 2, 3, 3, 4 })] // Higher-dimensional batched multiplication with same batch size

    // More complex cases combining broadcasting and batching
    [TestCase(new[] { 1, 2, 3 }, new[] { 5, 3, 4 })]
    [TestCase(new[] { 6, 1, 2, 3 }, new[] { 3, 5 })]
    [TestCase(new[] { 3, 1, 2, 3 }, new[] { 3, 8, 3, 4 })]
    [TestCase(new[] { 2, 3, 1, 3 }, new[] { 2, 1, 3, 4 })]

    // Corrected complex cases combining broadcasting and batching
    [TestCase(new[] { 5, 3, 4 }, new[] { 1, 4, 2 })]
    [TestCase(new[] { 6, 1, 2, 3 }, new[] { 1, 3, 4 })]
    [TestCase(new[] { 3, 2, 2, 3 }, new[] { 2, 3, 4 })]
    [TestCase(new[] { 2, 3, 4, 3 }, new[] { 2, 3, 3, 5 })]

    // Flipped dimensions for complex cases
    [TestCase(new[] { 1, 4, 3 }, new[] { 5, 3, 4 })]
    [TestCase(new[] { 5, 3, 4 }, new[] { 6, 1, 4, 3 })]
    [TestCase(new[] { 2, 3, 4 }, new[] { 3, 2, 4, 3 })]
    [TestCase(new[] { 2, 3, 3, 4 }, new[] { 2, 3, 4, 3 })]
    public void Compare_matmul_different_shapes(int[] a, int[] b)
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
             d = jnp.matmul(x, y)
             """);

        var x = scope.GetTensor("x");
        var y = scope.GetTensor("y");
        var d = scope.GetTensor("d");
        var z = Tensor.Matmul(x, y);

        TensorAsserts.ShapesAreEqual(d, z);
        TensorAsserts.ValuesAreEqual(d, z);
    }
}
