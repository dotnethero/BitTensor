using BitTensor.Abstractions;

namespace BitTensor.CUDA.Wrappers;

public sealed class CuRandContext : IDisposable
{
    internal readonly CuRandGenerator Generator = new(seed: 0);

    public CudaTensor<float> Uniform(Shape shape)
    {
        var tensor = new CudaTensor<float>(shape);
        Generator.GenerateUniform(tensor);
        return tensor;
    }
        
    public CudaTensor<float> Normal(Shape shape, float mean = 0f, float stddev = 1f)
    {
        using var generator = new CuRandGenerator();
        var tensor = new CudaTensor<float>(shape);
        Generator.GenerateNormal(tensor, mean, stddev);
        return tensor;
    }

    public void Dispose()
    {
        Generator.Dispose();
    }
}