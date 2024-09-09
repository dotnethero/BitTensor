using System.ComponentModel;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Models;

public sealed class ModelTrainer<T> where T : unmanaged, IFloatingPoint<T>
{
    public CudaDataset<T> InputDataset { get; }
    public CudaDataset<T> OutputDataset { get; }

    public ILayer<T> Model { get; }
    public CudaVariable<T> Inputs { get; }
    public CudaVariable<T> Desired { get; }
    public CudaNode<T> Outputs { get; }
    public CudaNode<T> Loss { get; }
    public GradientCollection<T> Gradients { get; }

    internal ModelTrainer(ILayer<T> model, LossFunction<T> lossFunction, CudaDataset<T> inputs, CudaDataset<T> outputs, int batchSize = 1)
    {
        Model = model;
        InputDataset = inputs;
        OutputDataset = outputs;

        var context = model.Context;
        var inputShape = inputs.Shape[1..];
        var outputShape = outputs.Shape[1..];

        Inputs = context.CreateVariable<T>([batchSize, ..inputShape]);
        Desired = context.CreateVariable<T>([batchSize, ..outputShape]);
        Outputs = model.Compose(Inputs);
        Loss = lossFunction(Outputs, Desired);
        Gradients = Loss.GetGradients();
    }

    public unsafe void Fit(float lr, int epochs, bool trace = false)
    {
        CUstream_st* stream = null;
        CUgraph_st* graph = null;
        CUgraphExec_st* graphInstance = null;

        var error0 = cudaRT.cudaStreamCreate(&stream);
        CuStream.Default = stream;

        for (var i = 0; i < epochs; i++)
        {
            var batchSize = Inputs.Shape[0];
            var datasetSize = InputDataset.Shape[0];
            var indexes = Enumerable.Range(0, datasetSize).ToArray();

            Random.Shared.Shuffle(indexes);

            for (var j = 0; j < datasetSize - batchSize; j += batchSize)
            {
                var batchIndexes = indexes.AsSpan(j, batchSize);
                Inputs.LoadBatches(InputDataset, batchIndexes);
                Desired.LoadBatches(OutputDataset, batchIndexes);

                if (j == 0 && graphInstance is null)
                {
                    Loss.EnsureHasUpdatedValues();
                    ApplyGradients(Model.Parameters, Gradients, lr);
                }

                if (j > 0 && graphInstance is null)
                {
                    var error1 = cudaRT.cudaStreamBeginCapture(stream, cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal);
                    Loss.EnsureHasUpdatedValues();
                    ApplyGradients(Model.Parameters, Gradients, lr);
                    var error2 = cudaRT.cudaStreamEndCapture(stream, &graph);
                    //var bytes = Encoding.ASCII.GetBytes(@"C:\Projects\graph.log");
                    //fixed (byte* b = bytes)
                    //{
                    //    var error25 = cudaRT.cudaGraphDebugDotPrint(graph, b, 0);
                    //}

                    var error3 = cudaRT.cudaGraphInstantiate(&graphInstance, graph, 1);
                }

                if (j > 0 && graphInstance is not null)
                {
                    var error5 = cudaRT.cudaGraphLaunch(graphInstance, stream);
                }
            }

            var error6 = cudaRT.cudaStreamSynchronize(stream);

            if (trace)
            {
                var loss = CuDebug.View(Loss);
                Console.WriteLine(loss);
            }
        }

        cudaRT.cudaStreamDestroy(stream);
        CuStream.Default = (CUstream_st*)0;
    }
    
    private static void ApplyGradients(CudaWeights<T>[] variables, GradientCollection<T> gradients, float lr)
    {
        foreach (var variable in variables)
        {
            var gradient = gradients.By(variable);
            gradient.EnsureHasUpdatedValues();
            variable.AdjustWeights(gradient.Tensor, lr);
        }
    }
}