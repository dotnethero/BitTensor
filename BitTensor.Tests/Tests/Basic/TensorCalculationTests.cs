using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using BitTensor.Core;
using BitTensor.Models;
using BitTensor.Units;
using NUnit.Framework;

namespace BitTensor.Tests.Basic;

[TestFixture]
[SuppressMessage("ReSharper", "InconsistentNaming")]
class TensorCalculationTests
{
    [Test]
    public static void Test_grad()
    {
        var a = Tensor.Create([1, 2, 3]);
        var b = Tensor.Create([3, 4, 5]);
        var c = Tensor.Create([-5, 0, 5]);
        var y = Tensor.Sum(a * a * b + c);

        var dy = Auto.Grad(y);

        var dyda = dy.By(a);
        var dydb = dy.By(b);
        var dydc = dy.By(c);
    }

    [Test]
    public static void Test_dot_product()
    {
        var a = Tensor.Create([1, 2, 3]);
        var b = Tensor.Create([3, 4, 5]);
        var c = a * b;
        var y = Tensor.Sum(c);

        var dy = Auto.Grad(y);

        var dyda = dy.By(a);
        var dydb = dy.By(b);
        var dydc = dy.By(c);
    }

    [Test]
    public static void Test_matrix_multiplication()
    {
        // Define two matrices
        var A = Tensor.Arrange(1, 5).Reshape([2, 2]);
        var B = Tensor.Arrange(5, 9).Reshape([2, 2]);

        // Perform matrix multiplication
        var C = Tensor.Matmul(A, B);  // Assuming MatMul is the method for matrix multiplication

        // Compute a function of the result, e.g, sum of all elements
        var Y = Tensor.Sum(C);

        // Compute gradients
        var dY = Auto.Grad(Y);

        var dYdA = dY.By(A);  // Gradient of Y with respect to A
        var dYdB = dY.By(B);  // Gradient of Y with respect to B
        var dYdC = dY.By(C);  // Gradient of Y with respect to C (the result of A * B)
        
        // Additional checks
        // Check the result of matrix multiplication
        Assert.That(C.ToDataString(), Is.EqualTo(
            """
            [[  19.00  22.00 ]
             [  43.00  50.00 ]]
            """));

        // Check the sum of all elements in C
        Assert.That(Y.ToDataString(), Is.EqualTo("134.00"));

        // Assertions for the gradients
        Assert.That(dYdA.ToDataString(), Is.EqualTo(
            """
            [[  11.00  15.00 ]
             [  11.00  15.00 ]]
            """));

        Assert.That(dYdB.ToDataString(), Is.EqualTo(
            """
            [[   4.00   4.00 ]
             [   6.00   6.00 ]]
            """));
    }
    
    [Test]
    public static void Test_matrix_multiplication_with_different_shapes()
    {
        // Define two matrices
        var A = Tensor.Create([
            [7, 3],
            [1, 2],
            [3, 4]]);

        var B = Tensor.Create([
            [5, 6, 1, -5], 
            [7, 8, -1, -2]]);

        var C = Tensor.Matmul(A, B);  // Assuming MatMul is the method for matrix multiplication
        var Y = Tensor.Sum(C);
        var dY = Auto.Grad(Y);

        var dYdA = dY.By(A);  // Gradient of Y with respect to A
        var dYdB = dY.By(B);  // Gradient of Y with respect to B
        
        Assert.That(C.ToDataString(), Is.EqualTo(
        """
        [[  56.00  66.00   4.00 -41.00 ]
         [  19.00  22.00  -1.00  -9.00 ]
         [  43.00  50.00  -1.00 -23.00 ]]
        """));

        // Assertions for the gradients
        Assert.That(dYdA.ToDataString(), Is.EqualTo(
            """
            [[   7.00  12.00 ]
             [   7.00  12.00 ]
             [   7.00  12.00 ]]
            """));

        Assert.That(dYdB.ToDataString(), Is.EqualTo(
            """
            [[  11.00  11.00  11.00  11.00 ]
             [   9.00   9.00   9.00   9.00 ]]
            """));
    }
    
    [Test]
    public static void Test_matrix_multiplication_3D()
    {
        // Define two matrices
        var A = Tensor.Create([[
            [7, 3],
            [1, 2],
            [3, 4]]]);

        var B = Tensor.Create([[
            [5, 6, 1, -5], 
            [7, 8, -1, -2]]]);

        var C = Tensor.Matmul(A, B);  // Assuming MatMul is the method for matrix multiplication
        var Y = Tensor.Sum(C);
        var dY = Auto.Grad(Y);

        var dYdA = dY.By(A);  // Gradient of Y with respect to A
        var dYdB = dY.By(B);  // Gradient of Y with respect to B
        
        Assert.That(C.ToDataString(), Is.EqualTo(
            """
            [[[  56.00  66.00   4.00 -41.00 ]
              [  19.00  22.00  -1.00  -9.00 ]
              [  43.00  50.00  -1.00 -23.00 ]]]
            """));

        // Assertions for the gradients
        Assert.That(dYdA.ToDataString(), Is.EqualTo(
            """
            [[[   7.00  12.00 ]
              [   7.00  12.00 ]
              [   7.00  12.00 ]]]
            """));

        Assert.That(dYdB.ToDataString(), Is.EqualTo(
            """
            [[[  11.00  11.00  11.00  11.00 ]
              [   9.00   9.00   9.00   9.00 ]]]
            """));
    }

    [Test]
    public static void Test_linear()
    {
        var x = Tensor.Create([1.0f, 0.0f, -1.0f]).Reshape([3, 1]);
        var b = Tensor.Create([-0.5f, -0.3f]).Reshape([2, 1]);

        var w = Tensor.Create([
            [.1f, .2f, .3f],
            [.2f, .1f, .4f]]);

        var z = Tensor.Matmul(w, x) + b;
        var y = Tensor.Sum(Tensor.Sigmoid(z));
        var dydw = Auto.Grad(y).By(w);

        Console.WriteLine($"{y.ToDataString()}");
        Console.WriteLine($"{dydw.ToDataString()}");
    }

    [Test]
    public static void Test_linear_unit()
    {
        var x = Tensor.Create([0.3f, 0.1f, -0.2f]).Reshape([1, 3]);
        var d = Tensor.Create([0.5f, 0.8f]).Reshape([1, 2]); // desired
        
        var unit = new LinearLayer(3, 2, Tensor.Sigmoid);
        var steps = 100_000;
        var lr = 0.001f;
        var y = unit.Compute(x);
        var diff = (y - d);
        var loss = Tensor.Sum(diff * diff) * 0.5f;
        var grad = Auto.Grad(loss);
        var gradients = grad(unit.Parameters);
        var sw = Stopwatch.StartNew();

        for (var i = 0; i < steps; ++i)
        {
            Auto.ApplyGradients(unit.Parameters, gradients, lr);
        }

        Console.WriteLine(sw.Elapsed);
    }

    [Test]
    public static void Test_linear_module()
    {
        const int inputCount = 1000;
        const int outputCount = 2;
        const int batchSize = 5;
        const int dataDimension = 1;

        var x = Tensor.Random.Normal([batchSize, inputCount]);
        var d = Tensor.Random.Normal([batchSize, outputCount]);
        var model = Model.Sequential(
        [
            new LinearLayer(x.Shape[dataDimension], d.Shape[dataDimension], Tensor.Sigmoid)
        ]);

        var compilation = model.Compile(x, d);
        var sw = Stopwatch.StartNew();
        model.Fit(compilation, lr: 0.001f, epochs: 100);
        Console.WriteLine(sw.Elapsed);
    }
    
    [Test]
    public void Test_linear_inv_module()
    {
        const int inputCount = 1000;
        const int outputCount = 2;
        const int batchSize = 5;
        const int dataDimension = 0;

        var x = Tensor.Random.Normal([batchSize, inputCount]).Transpose();
        var d = Tensor.Random.Normal([batchSize, outputCount]).Transpose();

        var model = Model.Sequential(
        [
            new LinearLayerInv(x.Shape[dataDimension], d.Shape[dataDimension], Tensor.Sigmoid)
        ]);

        var compilation = model.Compile(x, d);
        var sw = Stopwatch.StartNew();
        model.Fit(compilation, lr: 0.001f, epochs: 100);
        Console.WriteLine(sw.Elapsed);
    }
}
