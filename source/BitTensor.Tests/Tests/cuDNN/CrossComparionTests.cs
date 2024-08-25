using BitTensor.Core.Tests;
using BitTensor.CUDA;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Wrappers;
using NUnit.Framework;

namespace BitTensor.Tests.cuDNN;

[TestFixture]
internal class CrossComparionTests
{
    [Test]
    [TestCase(3, 8, 4)]
    [TestCase(1, 8, 4)]
    [TestCase(2, 1, 4)]
    [TestCase(2, 8, 2)] // TODO: fix random to generate 1 value
    public static void Test_linear_layer(int batchSize, int inputFeatures, int outputFeatures)
    {
        var random = new CuRandContext();

        using var inputs = random.Normal([batchSize, inputFeatures]);
        using var weights = random.Normal([inputFeatures, outputFeatures]);
        using var bias = random.Normal([outputFeatures]);
        using var outputs1 = new CudaTensor<float>([batchSize, outputFeatures]);
        using var outputs2 = new CudaTensor<float>([batchSize, outputFeatures]);

        using var cutensor = new CuTensorContext();
        using var cuplan1 = cutensor.CreateMatMulPlan<float>(inputs.Shape, weights.Shape, outputs1.Shape);
        using var cuplan2 = cutensor.CreateAddPlan<float>(outputs1.Shape, bias.Shape, outputs1.Shape);

        cuplan1.Execute(inputs, weights, outputs1);
        cuplan2.Execute(outputs1, bias, outputs1);

        CuDebug.WriteLine(outputs1);

        using var context = new CudnnContext();

        using var ti = inputs.CreateDescriptor();
        using var tw = weights.CreateDescriptor();
        using var tb = bias.CreateDescriptor();
        using var to = outputs2.CreateDescriptor();

        using var tt = Fusion.VirtualDescriptor<float>(outputs2.Shape);
        using var mm = Fusion.MatMul(ti, tw, tt);
        using var pw = Fusion.Add(tt, tb, to);

        using var graph = new CudnnGraph(context, [mm, pw]);
        using var pack = new CudnnVariantPack<float>([inputs, weights, bias, outputs2]);

        context.ExecuteGraph(graph, pack);
        CuDebug.WriteLine(outputs2);

        TensorAsserts.ValuesAreEqual(outputs1, outputs2, tolerance: 1e-3f);
    }
}