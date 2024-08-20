using BitTensor.Core.Tests;
using BitTensor.CUDA.Graph;
using NUnit.Framework;
using Python.Runtime;

namespace BitTensor.Tests.Comparison;

[TestFixture]
class UnaryComparisonTests
{
    [TestCase(new int[0])]
    [TestCase(new[] {4})]
    [TestCase(new[] {3, 4})]
    public void Compare_exp_and_gradients(int[] shape)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL();

        var pyshape = $"[{string.Join(",", shape)}]";

        scope.ExecuteJax(
            $"""
             def test_sum(A):
                result = jnp.exp(A)
                return jnp.sum(result)
                
             def test_dot(A):
                result = jnp.exp(A)
                return jnp.sum(jnp.multiply(result, result))
                
             key = jax.random.PRNGKey(0)
             a = jax.random.normal(key, {pyshape})
             d = jnp.exp(a)
             g = jax.grad(test_sum)(a)
             h = jax.grad(test_dot)(a)
             """);

        using var context = CudaContext.CreateDefault();
        
        var a = scope.GetTensor("a").AsNode(context);
        var d_true = scope.GetTensor("d").AsTensor();
        var g_true = scope.GetTensor("g").AsTensor();
        var h_true = scope.GetTensor("h").AsTensor();

        var z = Ops.Exp(a);
        var g = Ops.Sum(z).GetGradients().By(a);
        var h = Ops.DotProduct(z, z).GetGradients().By(a);

        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(d_true, z);
            TensorAsserts.ShapesAreEqual(g_true, g);
            TensorAsserts.ShapesAreEqual(h_true, h);
        });

        Assert.Multiple(() =>
        {
            TensorAsserts.ValuesAreEqual(d_true, z);
            TensorAsserts.ValuesAreEqual(g_true, g);
            TensorAsserts.ValuesAreEqual(h_true, h);
        });
    }

    [TestCase(new int[0])]
    [TestCase(new[] {4})]
    [TestCase(new[] {3, 4})]
    public void Compare_log_and_gradients(int[] shape)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL();

        var pyshape = $"[{string.Join(",", shape)}]";

        scope.ExecuteJax(
            $"""
             def test_sum(A):
                result = jnp.log(A)
                return jnp.sum(result)
                
             def test_dot(A):
                result = jnp.log(A)
                return jnp.sum(jnp.multiply(result, result))
                
             key = jax.random.PRNGKey(0)
             a = jnp.abs(jax.random.normal(key, {pyshape}))
             d = jnp.log(a)
             g = jax.grad(test_sum)(a)
             h = jax.grad(test_dot)(a)
             """);

        using var context = CudaContext.CreateDefault();
        
        var a = scope.GetTensor("a").AsNode(context);
        var d_true = scope.GetTensor("d").AsTensor();
        var g_true = scope.GetTensor("g").AsTensor();
        var h_true = scope.GetTensor("h").AsTensor();

        var z = Ops.Log(a);
        var g = Ops.Sum(z).GetGradients().By(a);
        var h = Ops.DotProduct(z, z).GetGradients().By(a);

        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(d_true, z);
            TensorAsserts.ShapesAreEqual(g_true, g);
            TensorAsserts.ShapesAreEqual(h_true, h);
        });

        Assert.Multiple(() =>
        {
            TensorAsserts.ValuesAreEqual(d_true, z);
            TensorAsserts.ValuesAreEqual(g_true, g);
            TensorAsserts.ValuesAreEqual(h_true, h);
        });
    }
    
    [TestCase(new int[0])]
    [TestCase(new[] {4})]
    [TestCase(new[] {3, 4})]
    public void Compare_reciprocal_and_gradients(int[] shape)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL();

        var pyshape = $"[{string.Join(",", shape)}]";

        scope.ExecuteJax(
            $"""
             def test_sum(A):
                result = 1 / A
                return jnp.sum(result)
                
             def test_dot(A):
                result = 1 / A
                return jnp.sum(jnp.multiply(result, result))
                
             key = jax.random.PRNGKey(0)
             a = jax.random.normal(key, {pyshape})
             d = 1 / a
             g = jax.grad(test_sum)(a)
             h = jax.grad(test_dot)(a)
             """);

        using var context = CudaContext.CreateDefault();
        
        var a = scope.GetTensor("a").AsNode(context);
        var d_true = scope.GetTensor("d").AsTensor();
        var g_true = scope.GetTensor("g").AsTensor();
        var h_true = scope.GetTensor("h").AsTensor();

        var z = Ops.Reciprocal(a);
        var g = Ops.Sum(z).GetGradients().By(a);
        var h = Ops.DotProduct(z, z).GetGradients().By(a);

        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(d_true, z);
            TensorAsserts.ShapesAreEqual(g_true, g);
            TensorAsserts.ShapesAreEqual(h_true, h);
        });

        Assert.Multiple(() =>
        {
            TensorAsserts.ValuesAreEqual(d_true, z);
            TensorAsserts.ValuesAreEqual(g_true, g);
            TensorAsserts.ValuesAreEqual(h_true, h);
        });
    }
}