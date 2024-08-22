using BitTensor.Core.Tests;
using BitTensor.CUDA.Graph;
using NUnit.Framework;
using Python.Runtime;

namespace BitTensor.Tests.Comparison;

[TestFixture]
class BinaryComparisonTests
{
    [TestCase(new int[0])]
    [TestCase(new[] {4})]
    [TestCase(new[] {3, 4})]
    public void Compare_multiplication_and_gradient(int[] shape)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL();

        var pyshape = $"[{string.Join(",", shape)}]";

        scope.ExecuteJax(
            $"""
             def test_sum(A, B):
                result = jnp.multiply(A, B)
                return jnp.sum(result)
                
             key = jax.random.PRNGKey(0)
             a = jax.random.normal(key, {pyshape})
             b = jax.random.normal(key, {pyshape})
             z = jnp.multiply(a, b)
             da = jax.grad(test_sum, argnums=0)(a, b)
             db = jax.grad(test_sum, argnums=1)(a, b)
             """);

        using var context = CudaContext.CreateDefault();
        
        var a = scope.GetTensor("a").AsNode(context);
        var b = scope.GetTensor("b").AsNode(context);
        var z_true = scope.GetTensor("z").AsTensor();
        var da_true = scope.GetTensor("da").AsTensor();
        var db_true = scope.GetTensor("db").AsTensor();

        var z = a * b;
        var dz = Ops.Sum(z).GetGradients();
        var da = dz.By(a);
        var db = dz.By(b);

        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(z_true, z);
            TensorAsserts.ShapesAreEqual(da_true, da);
            TensorAsserts.ShapesAreEqual(db_true, db);
        });

        Assert.Multiple(() =>
        {
            TensorAsserts.ValuesAreEqual(z_true, z);
            TensorAsserts.ValuesAreEqual(da_true, da);
            TensorAsserts.ValuesAreEqual(db_true, db);
        });
    }

    [TestCase(new int[0])]
    [TestCase(new[] {4})]
    [TestCase(new[] {3, 4})]
    public void Compare_division_and_gradient(int[] shape)
    {
        using var scope = Py.CreateScope();
        using var _ = Py.GIL();

        var pyshape = $"[{string.Join(",", shape)}]";

        scope.ExecuteJax(
            $"""
             def test_sum(A, B):
                result = A / B
                return jnp.sum(result)
                
             key = jax.random.PRNGKey(0)
             a = jax.random.normal(key, {pyshape})
             b = jax.random.normal(key, {pyshape})
             z = a / b
             da = jax.grad(test_sum, argnums=0)(a, b)
             db = jax.grad(test_sum, argnums=1)(a, b)
             """);

        using var context = CudaContext.CreateDefault();
        
        var a = scope.GetTensor("a").AsNode(context);
        var b = scope.GetTensor("b").AsNode(context);
        var z_true = scope.GetTensor("z").AsTensor();
        var da_true = scope.GetTensor("da").AsTensor();
        var db_true = scope.GetTensor("db").AsTensor();

        var z = a / b;
        var dz = Ops.Sum(z).GetGradients();
        var da = dz.By(a);
        var db = dz.By(b);

        Assert.Multiple(() =>
        {
            TensorAsserts.ShapesAreEqual(z_true, z);
            TensorAsserts.ShapesAreEqual(da_true, da);
            TensorAsserts.ShapesAreEqual(db_true, db);
        });

        Assert.Multiple(() =>
        {
            TensorAsserts.ValuesAreEqual(z_true, z);
            TensorAsserts.ValuesAreEqual(da_true, da);
            TensorAsserts.ValuesAreEqual(db_true, db);
        });
    }

}