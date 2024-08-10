using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public sealed class SequentialModel(ILayer[] layers) : Model
{
    public override CuTensorWeights[] Parameters =>  
        layers.SelectMany(x => x.Parameters).ToArray();

    public override CuTensorNode Compute(CuTensorNode input) => 
        layers.Aggregate(input, (activation, layer) => layer.Compute(activation));
}