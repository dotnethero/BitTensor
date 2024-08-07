using BitTensor.Core;
using BitTensor.Units;

namespace BitTensor.Models;

public sealed class SequentialModel(ILayer[] layers) : Model
{
    public override Tensor[] Parameters =>  
        layers.SelectMany(x => x.Parameters).ToArray();

    public override Tensor Compute(Tensor input) => 
        layers.Aggregate(input, (activation, layer) => layer.Compute(activation));
}
