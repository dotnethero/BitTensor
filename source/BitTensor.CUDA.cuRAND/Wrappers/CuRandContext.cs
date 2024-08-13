using BitTensor.Abstractions;

namespace BitTensor.CUDA.Wrappers;

public sealed class CuRandContext
{
    public CudaTensor<float> Uniform(Shape shape)
    {
        using var generator = new CuRandGenerator(seed: 0);
        var tensor = new CudaTensor<float>(shape);
        generator.GenerateUniform(tensor);
        return tensor;
    }
        
    public CudaTensor<float> Normal(Shape shape, float mean = 0f, float stddev = 1f)
    {
        using var generator = new CuRandGenerator(seed: 0);
        var tensor = new CudaTensor<float>(shape);
        generator.GenerateNormal(tensor, mean, stddev);
        return tensor;
    }
}