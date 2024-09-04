using BitTensor.Core.Tests;
using BitTensor.CUDA;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;
using NUnit.Framework;

namespace BitTensor.Tests.cuDNN;

[TestFixture]
internal class CrossComparionTests
{
    [SetUp]
    public void SetEnvironment()
    {
        Environment.SetEnvironmentVariable("CUDNN_LOGDEST_DBG", "stdout");
        Environment.SetEnvironmentVariable("CUDNN_LOGLEVEL_DBG", "2");
    }
    
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

        using var tt = Fusion.CreateVirtualDescriptor<float>(outputs2.Shape);
        using var mm = Fusion.MatMul(ti, tw, tt);
        using var pw = Fusion.Add(tt, tb, to);

        using var graph = new CudnnGraph(context, [mm, pw]);
        using var pack = new CudnnVariantPack<float>([inputs, weights, bias, outputs2]);

        using var plan = graph.GetExecutionPlan();

        plan.Execute(pack);
        CuDebug.WriteLine(outputs2);

        TensorAsserts.ValuesAreEqual(outputs1, outputs2, tolerance: 1e-3f);
    }

    [Test]
    public static void Test_reduction()
    {
        var random = new CuRandContext();
        
        using var inputs = random.Normal([8, 16]);
        using var outputs1 = new CudaTensor<float>([8, 1]);
        using var outputs2 = new CudaTensor<float>([8, 1]);
        
        using var cutensor = new CuTensorContext();
        using var cuplan = cutensor.CreateReductionPlan<float>(
            inputs.Shape,
            outputs1.Shape,
            axis: [^1],
            operation: cutensorOperator_t.CUTENSOR_OP_ADD,
            keepDims: true);

        cuplan.Execute(inputs, outputs1);

        CuDebug.WriteLine(outputs1);
        
        using var context = new CudnnContext();

        using var ti = inputs.CreateDescriptor();
        using var to = outputs2.CreateDescriptor();

        using var sum = Fusion.Sum(ti, to);
        using var graph = new CudnnGraph(context, [sum]);
        using var plan = graph.GetExecutionPlan();
        using var pack = new CudnnVariantPack<float>([inputs, outputs2]);
        
        plan.Execute(pack);
        
        CuDebug.WriteLine(outputs2);
        
        TensorAsserts.ValuesAreEqual(outputs1, outputs2, tolerance: 1e-3f);
    }
}