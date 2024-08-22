using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public static class Model
{
    public static Sequential<T> Create<T>(
        ILayer<T>[] layers) 
        where T : unmanaged, IFloatingPoint<T> => 
        new(layers);

    public static ModelTrainer<T> Compile<T>(
        ILayer<T> model,
        LossFunction<T> lossFunction,
        Dataset<T> inputs,
        Dataset<T> outputs,
        int batchSize = 1)
        where T : unmanaged, IFloatingPoint<T> => 
        new(model, lossFunction, inputs, outputs, batchSize);
}