using BitTensor.Core.Tests;
using BitTensor.CUDA.Graph;
using NUnit.Framework;
using Python.Runtime;

namespace BitTensor.Tests.Comparison;

[TestFixture]
[NonParallelizable]
class GemmGradientTests
{
    [Test]
    [TestCase(new[] { 1 }, new[] { 1 })] // Dot product
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
    public void Compare_gemm_operation(int[] ai, int[] bi)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL(); 
        
        var a_shape = $"[{string.Join(",", ai)}]";
        var b_shape = $"[{string.Join(",", bi)}]";
        
        scope.ExecuteJax(
            $"""
             key = jax.random.PRNGKey(0)
             ak, bk, ck = jax.random.split(key, 3)
             a = jax.random.normal(ak, {a_shape})
             b = jax.random.normal(bk, {b_shape})
             c = jax.random.normal(ck, jnp.matmul(a, b).shape)
             z = jnp.matmul(a, b) + c
             """);

        using var context = CudaContext.CreateDefault();
        var a = scope.GetTensor("a").AsNode(context);
        var b = scope.GetTensor("b").AsNode(context);
        var c = scope.GetTensor("c").AsNode(context);

        var z = Ops.Gemm(a, b, c);
        var z_true = scope.GetTensor("z").AsTensor();

        context.Synchronize();

        TensorAsserts.ShapesAreEqual(z_true, z);
        TensorAsserts.ValuesAreEqual(z_true, z);
    }

    [Test]
    [TestCase(new[] { 1 }, new[] { 1 })] // Dot product
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
    public void Compare_gemm_operation_gradients(int[] ai, int[] bi)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL(); 
        
        var a_shape = $"[{string.Join(",", ai)}]";
        var b_shape = $"[{string.Join(",", bi)}]";
        
        scope.ExecuteJax(
            $"""
             def gemm(A, B, C):
                return jnp.sum(jnp.matmul(A, B) + C)
             
             key = jax.random.PRNGKey(0)
             ak, bk, ck = jax.random.split(key, 3)
             a = jax.random.normal(ak, {a_shape})
             b = jax.random.normal(bk, {b_shape})
             c = jax.random.normal(ck, jnp.matmul(a, b).shape)
             z = jnp.matmul(a, b) + c
             
             da = jax.grad(gemm, argnums=0)(a, b, c)
             db = jax.grad(gemm, argnums=1)(a, b, c)
             dc = jax.grad(gemm, argnums=2)(a, b, c)
             """);

        using var context = CudaContext.CreateDefault();
        var a = scope.GetTensor("a").AsNode(context);
        var b = scope.GetTensor("b").AsNode(context);
        var c = scope.GetTensor("c").AsNode(context);

        var z = Ops.Gemm(a, b, c);
        var z_true = scope.GetTensor("z").AsTensor();

        var da_true = scope.GetTensor("da").AsTensor();
        var db_true = scope.GetTensor("db").AsTensor();
        var dc_true = scope.GetTensor("dc").AsTensor();

        var grads = Ops.Sum(z).GetGradients();
        var da = grads[a];
        var db = grads[b];
        var dc = grads[c];

        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(z_true, z);
            TensorAsserts.ShapesAreEqual(da_true, da);
            TensorAsserts.ShapesAreEqual(db_true, db);
            TensorAsserts.ShapesAreEqual(dc_true, dc);
        });

        Assert.Multiple(() => 
        {
            TensorAsserts.ValuesAreEqual(z_true, z);
            TensorAsserts.ValuesAreEqual(da_true, da);
            TensorAsserts.ValuesAreEqual(db_true, db);
            TensorAsserts.ValuesAreEqual(dc_true, dc);
        });
    }
}
