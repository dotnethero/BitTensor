using System.Numerics;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public sealed class SequentialModel<T>(ILayer<T>[] layers) : Model<T> where T : unmanaged, IFloatingPoint<T>
{
    public override CudaWeights<T>[] Parameters =>  
        layers.SelectMany(x => x.Parameters).ToArray();

    public override CudaNode<T> Compute(CudaNode<T> input) => 
        layers.Aggregate(input, (activation, layer) => layer.Compute(activation));
}