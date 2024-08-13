using BitTensor.Abstractions;

namespace BitTensor.CUDA.Wrappers;

public sealed class CuRandContext
{
    public CuTensor<float> Uniform(Shape shape)
    {
        using var generator = new CuRandGenerator(seed: 0);
        var tensor = new CuTensor<float>(shape);
        generator.GenerateUniform(tensor);
        return tensor;
    }
        
    public CuTensor<float> Normal(Shape shape, float mean = 0f, float stddev = 1f)
    {
        using var generator = new CuRandGenerator(seed: 0);
        var tensor = new CuTensor<float>(shape);
        generator.GenerateNormal(tensor, mean, stddev);
        return tensor;
    }
}