using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA;

public partial class CuTensor
{
    public static class Random
    {
        public static CuTensor Uniform(Shape shape)
        {
            using var generator = new CuRandGenerator(seed: 0);
            var tensor = new CuTensor(shape);
            generator.GenerateUniform(tensor);
            return tensor;
        }
        
        public static CuTensor Normal(Shape shape, float mean = 0f, float stddev = 1f)
        {
            using var generator = new CuRandGenerator(seed: 0);
            var tensor = new CuTensor(shape);
            generator.GenerateNormal(tensor, mean, stddev);
            return tensor;
        }
    }
}
