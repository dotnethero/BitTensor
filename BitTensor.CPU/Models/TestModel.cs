using BitTensor.Core;
using BitTensor.Units;

namespace BitTensor.Models;

public class TestModel(int inputs, int hidden, int outputs) : Model
{
    private readonly LinearLayer _input = new(inputs, hidden, Tensor.Tanh);
    private readonly LinearLayer _hidden = new(hidden, outputs, Tensor.Identity);

    public override Tensor[] Parameters => [.._input.Parameters, .._hidden.Parameters];

    public override Tensor Compute(Tensor input)
    {
        var activation = input;
        var layers = new[] { _input, _hidden };
        foreach (var layer in layers)
        {
            activation = layer.Compute(activation);
        }
        return activation;
    }
}