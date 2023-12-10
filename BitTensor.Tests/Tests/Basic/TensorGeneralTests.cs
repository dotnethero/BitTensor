﻿using BitTensor.Core;
using BitTensor.Models;
using NUnit.Framework;

namespace BitTensor.Tests.Basic;

[TestFixture]
class TensorGeneralTests
{
    [Test]
    public static void Total_tensor_count_should_match_baseline()
    {
        Tensor.MaxID = 0;

        var x = Tensor.Random.Normal([1000, 4]);
        var d = Tensor.Random.Normal([1000, 1]);

        var model = new TestModel(x.Shape[1], 7, d.Shape[1]);
        var compilation = model.Compile(x, d);

        Assert.That(Tensor.MaxID, Is.EqualTo(50));

        model.Fit(compilation, lr: 1e-2f, epochs: 1000);

        Assert.That(Tensor.MaxID, Is.EqualTo(50));
    }

    [Test]
    public static void Should_train_XOR_to_baseline_accuracy() // TODO: add baseline
    {
        var x = Tensor.Create([[0, 0], [0, 1], [1, 0], [1, 1]]);
        var d = Tensor.Create([0, 1, 1, 0]).Reshape([4, 1]);

        var model = new TestModel(2, 7, 1);

        var test1 = model.Compute(x).ToDataString();

        Console.WriteLine(test1);

        var compilation = model.Compile(x, d);
        model.Fit(compilation, lr: 1e-2f, epochs: 2000, trace: true);
        
        var test2 = model.Compute(x).ToDataString();

        Console.WriteLine(test2);

        Assert.That(compilation.Loss.Values.Scalar(), Is.LessThan(1e-4f));
    }

}
