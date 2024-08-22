using System.Numerics;
using BitTensor.CUDA.Graph;

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

    public void Fit(float lr, int epochs, bool trace = false)
    {
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
                Loss.EnsureHasUpdatedValues();
                ApplyGradients(Model.Parameters, Gradients, lr);
            }

            var loss = CuDebug.View(Loss);
            Console.WriteLine(loss);
        }
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