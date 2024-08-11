using BitTensor.Core.Tests;
using BitTensor.CUDA;
using BitTensor.CUDA.Graph;
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

        using var context = new CuContext();
        var x = scope.Get2D("x").AsNode(context);
        var y = scope.Get2D("y").AsNode(context);

        var xg_true = scope.Get2D("xg").AsTensor(context);
        var yg_true = scope.Get2D("yg").AsTensor(context);

        var grads = CuTensorNode.Sum(x * y).GetGradients();
        var xg = grads[x];
        var yg = grads[y];

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

        Console.WriteLine(ab_da_shape);
        Console.WriteLine(ab_db_shape);
        Console.WriteLine(ca_dc_shape);
        Console.WriteLine(ca_da_shape);
        
        using var context = new CuContext();

        var ab_da_true = scope.Get1D("ab_da").AsTensor(context);
        var ab_db_true = scope.Get2D("ab_db").AsTensor(context);
        var ca_dc_true = scope.Get2D("ca_dc").AsTensor(context);
        var ca_da_true = scope.Get1D("ca_da").AsTensor(context);

        var a = scope.Get1D("a").AsNode(context);
        var b = scope.Get2D("b").AsNode(context);
        var c = scope.Get2D("c").AsNode(context);
        
        var ab_grads = CuTensorNode.Sum(a * b).GetGradients();
        var ab_da = ab_grads[a];
        var ab_db = ab_grads[b];
        
        var ca_grads = CuTensorNode.Sum(c * a).GetGradients();
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

        Console.WriteLine(aс_da_shape);
        Console.WriteLine(aс_dс_shape);
        Console.WriteLine(cb_dc_shape);
        Console.WriteLine(cb_db_shape);

        using var context = new CuContext();

        var aс_da_true = scope.Get1D("aс_da").AsTensor(context);
        var aс_dс_true = scope.Get2D("aс_dс").AsTensor(context);
        var cb_dc_true = scope.Get2D("cb_dc").AsTensor(context);
        var cb_db_true = scope.Get1D("cb_db").AsTensor(context);

        var a = scope.Get1D("a").AsNode(context);
        var b = scope.Get1D("b").AsNode(context);
        var c = scope.Get2D("c").AsNode(context);

        var aс_grads = CuTensorNode.Sum(a * c).GetGradients();
        var aс_da = aс_grads[a];
        var ac_dc = aс_grads[c];
        
        var cb_grads = CuTensorNode.Sum(c * b).GetGradients();
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
    
    [Test]

    // Scalars
    [TestCase(new int[0], new int[0])] // Scalar x Scalar
    [TestCase(new int[0], new[] { 3 })] // Scalar x Vector
    [TestCase(new int[0], new[] { 3, 2 })] // Scalar x Matrix
    [TestCase(new int[0], new[] { 4, 3, 2 })] // Scalar x Batched matrix
    [TestCase(new[] { 3 }, new int[0])] // Vector x Scalar
    [TestCase(new[] { 3, 2 }, new int[0])] // Matrix x Scalar
    [TestCase(new[] { 4, 3, 2 }, new int[0])] // Batched matrix x Scalar

    [TestCase(new[] { 1 }, new[] { 1 })] // Scalar multiplication
    [TestCase(new[] { 3 }, new[] { 3 })] // Dot product
    [TestCase(new[] { 2, 3 }, new[] { 3, 4 })] // Regular matrix multiplication

    // Vector and matrix multiplication
    [TestCase(new[] { 3 }, new[] { 3, 2 })]    // Vector and matrix
    [TestCase(new[] { 3 }, new[] { 5, 3, 2 })] // Vector and matrix batch
    [TestCase(new[] { 2, 3 }, new[] { 3 })]    // Matrix and vector
    [TestCase(new[] { 5, 2, 3 }, new[] { 3 })] // Matrix batch and vector

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
    public void Compare_matmul_operation_gradients(int[] a, int[] b)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL(); 
        
        var x_py_shape = $"[{string.Join(",", a)}]";
        var y_py_shape = $"[{string.Join(",", b)}]";
        
        var function = a.Length == 0 || b.Length == 0 
            ? "multiply" 
            : "matmul";

        scope.ExecuteJax(
            $"""
             def func(A, B):
                return jnp.sum(jnp.{function}(A, B))
             
             key = jax.random.PRNGKey(0)
             xk, yk = jax.random.split(key)
             x = jax.random.normal(xk, {x_py_shape})
             y = jax.random.normal(yk, {y_py_shape})
             
             xy_dx = jax.grad(func, argnums=0)(x, y)
             xy_dy = jax.grad(func, argnums=1)(x, y)
             """);

        using var context = new CuContext();
        var x = scope.GetTensor("x").AsNode(context);
        var y = scope.GetTensor("y").AsNode(context);
        var z = x * y;

        var xy_dx_true = scope.GetTensor("xy_dx").AsTensor(context);
        var xy_dy_true = scope.GetTensor("xy_dy").AsTensor(context);
        
        var xy_grads = CuTensorNode.Sum(z).GetGradients();
        var xy_dx = xy_grads[x];
        var xy_dy = xy_grads[y];

        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(xy_dx_true, xy_dx);
            TensorAsserts.ShapesAreEqual(xy_dy_true, xy_dy);
        });

        Assert.Multiple(() => 
        {
            TensorAsserts.ValuesAreEqual(xy_dx_true, xy_dx);
            TensorAsserts.ValuesAreEqual(xy_dy_true, xy_dy);
        });
    }
}
