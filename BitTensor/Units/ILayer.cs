using BitTensor.Core;

namespace BitTensor.Units;

public interface ILayer
{
    Tensor[] Parameters { get; }
    Tensor Compute(Tensor input);
}
