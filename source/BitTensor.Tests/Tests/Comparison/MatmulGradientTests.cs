﻿using BitTensor.Abstractions;
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

        var x = scope.Get2D("x");
        var y = scope.Get2D("y");
        var xg_true = scope.Get2D("xg");
        var yg_true = scope.Get2D("yg");

        var x_node = new CuTensorNode(x);
        var y_node = new CuTensorNode(y);
        var grads = CuTensorNode.Sum(x_node * y_node).GetGradients();
        var xg = grads[x_node];
        var yg = grads[y_node];

        CuDebug.WriteLine(xg);
        CuDebug.WriteLine(yg);

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
        
        var a_node = new CuTensorNode(a);
        var b_node = new CuTensorNode(b);
        var c_node = new CuTensorNode(c);

        var ab_grads = CuTensorNode.Sum(a_node * b_node).GetGradients();
        var ab_da = ab_grads[a_node];
        var ab_db = ab_grads[b_node];
        
        var ca_grads = CuTensorNode.Sum(c_node * a_node).GetGradients();
        var ca_dc = ca_grads[c_node];
        var ca_da = ca_grads[a_node];

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

        var a_node = new CuTensorNode(a);
        var b_node = new CuTensorNode(b);
        var c_node = new CuTensorNode(c);

        var aс_grads = CuTensorNode.Sum(a_node * c_node).GetGradients();
        var aс_da = aс_grads[a_node];
        var ac_dc = aс_grads[c_node];
        
        var cb_grads = CuTensorNode.Sum(c_node * b_node).GetGradients();
        var cb_dc = cb_grads[c_node];
        var cb_db = cb_grads[b_node];

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
    public void Compare_matmul_operation_gradients(int[] a, int[] b)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL(); 
        
        var x_py_shape = $"[{string.Join(",", a)}]";
        var y_py_shape = $"[{string.Join(",", b)}]";

        scope.ExecuteJax(
            $"""
             def func(A, B):
                return jnp.sum(jnp.matmul(A, B))
             
             key = jax.random.PRNGKey(0)
             xk, yk = jax.random.split(key)
             x = jax.random.normal(xk, {x_py_shape})
             y = jax.random.normal(yk, {y_py_shape})
             
             xy_dx = jax.grad(func, argnums=0)(x, y)
             xy_dy = jax.grad(func, argnums=1)(x, y)
             """);


        var x = scope.GetTensor("x");
        var y = scope.GetTensor("y");
        
        var x_node = new CuTensorNode(x);
        var y_node = new CuTensorNode(y);
        var z_node = x_node * y_node;

        var xy_dx_true = scope.GetTensor("xy_dx");
        var xy_dy_true = scope.GetTensor("xy_dy");
        
        var xy_grads = CuTensorNode.Sum(z_node).GetGradients();
        var xy_dx = xy_grads[x_node];
        var xy_dy = xy_grads[y_node];
        
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