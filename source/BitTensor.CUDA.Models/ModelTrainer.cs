using System.Numerics;
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
        using var stream = new CudaStream();
        
        CudaGraph graph = null;
        CudaGraphInstance graphInstance = null;
        CuStream.Default = stream.Pointer; // TODO: Update operation contracts

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
                    // perform allocations on first run
                    Loss.EnsureHasUpdatedValues();
                    ApplyGradients(Model.Parameters, Gradients, lr);
                }

                if (j > 0 && graphInstance is null)
                {
                    // capture execution graph
                    stream.BeginCapture();
                    Loss.EnsureHasUpdatedValues();
                    ApplyGradients(Model.Parameters, Gradients, lr);
                    graph = stream.EndCapture();
                    graphInstance = graph.CreateInstance();
                }

                if (j > 0 && graphInstance is not null)
                {
                    graphInstance.Launch(stream);
                }
            }
            
            stream.Synchronize();
            
            if (trace)
            {
                var loss = CuDebug.View(Loss);
                Console.WriteLine(loss);
            }
        }

        graphInstance?.Dispose();
        graph?.Dispose();
        
        CuStream.Default = default;
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