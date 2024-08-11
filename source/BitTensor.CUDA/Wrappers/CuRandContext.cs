using BitTensor.Abstractions;

namespace BitTensor.CUDA.Wrappers;

public sealed class CuRandContext(CuContext context)
{
    public CuTensor Uniform(Shape shape)
    {
        using var generator = new CuRandGenerator(seed: 0);
        var tensor = context.Allocate(shape);
        generator.GenerateUniform(tensor);
        return tensor;
    }
        
    public CuTensor Normal(Shape shape, float mean = 0f, float stddev = 1f)
    {
        using var generator = new CuRandGenerator(seed: 0);
        var tensor = context.Allocate(shape);
        generator.GenerateNormal(tensor, mean, stddev);
        return tensor;
    }
}