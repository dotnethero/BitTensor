using BitTensor.Core;
using BitTensor.Core.Tests;
using NUnit.Framework;
using Python.Runtime;

namespace BitTensor.Tests.Comparison;

[TestFixture]
class MatmulGradientTests
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
    public void Compare_2Dx2D_matmul_gradients(int a, int b, int c)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL();

        scope.ExecuteJax(
            $"""
            def func(A, B):
                return jnp.sum(jnp.dot(A, B))
                
            key = jax.random.PRNGKey(0)
            x = jax.random.normal(key, [{a}, {b}])
            y = jax.random.normal(key, [{b}, {c}])

            xg = jax.grad(func, argnums=0)(x, y)
            yg = jax.grad(func, argnums=1)(x, y)
            """);

        var x = scope.Get2D("x");;
        var y = scope.Get2D("y");
        var xg_true = scope.Get2D("xg");
        var yg_true = scope.Get2D("yg");

        var grads = Auto.GetGradients(Tensor.Sum(Tensor.Matmul(x, y)));
        var xg = grads[x];
        var yg = grads[y];

        Console.WriteLine(xg.ToDataString());
        Console.WriteLine(yg.ToDataString());

        TensorAsserts.ShapesAreEqual(xg_true, xg);
        TensorAsserts.ValuesAreEqual(yg_true, yg);
    }

    [Test]
    [TestCase(1, 1)]
    [TestCase(3, 1)]
    [TestCase(1, 12)]
    [TestCase(3, 12)]
    public void Compare_2Dx1Dx2D_matmul_gradients(int n, int m)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL();

        scope.ExecuteJax(
            $"""
             def func(A, B):
                 return jnp.sum(jnp.dot(A, B))
             
             key = jax.random.PRNGKey(0)
             a = jax.random.normal(key, [{n}])
             b = jax.random.normal(key, [{n}, {m}])
             c = jax.random.normal(key, [{m}, {n}])
             
             ab_da = jax.grad(func, argnums=0)(a, b)
             ab_db = jax.grad(func, argnums=1)(a, b)
             
             ca_dc = jax.grad(func, argnums=0)(c, a)
             ca_da = jax.grad(func, argnums=1)(c, a)
             """);

        var ab_da_shape = scope.GetShape("ab_da");
        var ab_db_shape = scope.GetShape("ab_db");
        var ca_dc_shape = scope.GetShape("ca_dc");
        var ca_da_shape = scope.GetShape("ca_da");

        Console.WriteLine(ab_da_shape.Serialize());
        Console.WriteLine(ab_db_shape.Serialize());
        Console.WriteLine(ca_dc_shape.Serialize());
        Console.WriteLine(ca_da_shape.Serialize());
        
        var ab_da_true = scope.Get1D("ab_da");
        var ab_db_true = scope.Get2D("ab_db");
        var ca_dc_true = scope.Get2D("ca_dc");
        var ca_da_true = scope.Get1D("ca_da");

        var a = scope.Get1D("a");
        var b = scope.Get2D("b");
        var c = scope.Get2D("c");

        var ab_grads = Auto.GetGradients(Tensor.Sum(Tensor.Matmul(a, b)));
        var ab_da = ab_grads[a];
        var ab_db = ab_grads[b];
        
        var ca_grads = Auto.GetGradients(Tensor.Sum(Tensor.Matmul(c, a)));
        var ca_dc = ca_grads[c];
        var ca_da = ca_grads[a];

        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(ab_da_true, ab_da);
            TensorAsserts.ShapesAreEqual(ab_db_true, ab_db);
            TensorAsserts.ShapesAreEqual(ca_dc_true, ca_dc);
            TensorAsserts.ShapesAreEqual(ca_da_true, ca_da);
        });

        Assert.Multiple(() => 
        {
            TensorAsserts.ValuesAreEqual(ab_da_true, ab_da);
            TensorAsserts.ValuesAreEqual(ab_db_true, ab_db); // actual: ca_dc
            TensorAsserts.ValuesAreEqual(ca_dc_true, ca_dc); // actual: ab_db
            TensorAsserts.ValuesAreEqual(ca_da_true, ca_da);
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
             def func(A, B):
                return jnp.sum(jnp.dot(A, B))
             
             key = jax.random.PRNGKey(0)
             a = jax.random.normal(key, [{n}])
             b = jax.random.normal(key, [{m}])
             c = jax.random.normal(key, [{n}, {m}])

             aс_da = jax.grad(func, argnums=0)(a, c)
             aс_dс = jax.grad(func, argnums=1)(a, c)
             cb_dc = jax.grad(func, argnums=0)(c, b)
             cb_db = jax.grad(func, argnums=1)(c, b)
             """);
        
        var aс_da_shape = scope.GetShape("aс_da");
        var aс_dс_shape = scope.GetShape("aс_dс");
        var cb_dc_shape = scope.GetShape("cb_dc");
        var cb_db_shape = scope.GetShape("cb_db");

        Console.WriteLine(aс_da_shape.Serialize());
        Console.WriteLine(aс_dс_shape.Serialize());
        Console.WriteLine(cb_dc_shape.Serialize());
        Console.WriteLine(cb_db_shape.Serialize());

        var aс_da_true = scope.Get1D("aс_da");
        var aс_dс_true = scope.Get2D("aс_dс");
        var cb_dc_true = scope.Get2D("cb_dc");
        var cb_db_true = scope.Get1D("cb_db");

        var a = scope.Get1D("a");
        var b = scope.Get1D("b");
        var c = scope.Get2D("c");

        var aс_grads = Auto.GetGradients(Tensor.Sum(Tensor.Matmul(a, c)));
        var aс_da = aс_grads[a];
        var ac_dc = aс_grads[c];
        
        var cb_grads = Auto.GetGradients(Tensor.Sum(Tensor.Matmul(c, b)));
        var cb_dc = cb_grads[c];
        var cb_db = cb_grads[b];

        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(aс_da_true, aс_da);
            TensorAsserts.ShapesAreEqual(aс_dс_true, ac_dc);
            TensorAsserts.ShapesAreEqual(cb_dc_true, cb_dc);
            TensorAsserts.ShapesAreEqual(cb_db_true, cb_db);
        });

        Assert.Multiple(() => 
        {
            TensorAsserts.ValuesAreEqual(aс_da_true, aс_da);
            TensorAsserts.ValuesAreEqual(aс_dс_true, ac_dc);
            TensorAsserts.ValuesAreEqual(cb_dc_true, cb_dc);
            TensorAsserts.ValuesAreEqual(cb_db_true, cb_db);
        });
    }
}
