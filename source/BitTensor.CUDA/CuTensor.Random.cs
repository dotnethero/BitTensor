using BitTensor.Abstractions;

namespace BitTensor.CUDA;

public partial class CuTensor
{
    static System.Random Debug = new(0);

    public static class Random // TODO: use cuRAND
    {
        public static CuTensor Uniform(Shape shape, float min = -1f, float max = +1f)
        {
            var tensor = new CuTensor(shape);
            var values = new float[tensor.Size];

            for (var i = 0; i < values.Length; i++)
            {
                values[i] = NextUniform(min, max);
            }

            tensor.CopyToDevice(values);
            return tensor;
        }
        
        private static float NextUniform(float min = -1f, float max = 1f)
        {
            return Debug.NextSingle() * (max - min) + min;
        }

        public static CuTensor Normal(Shape shape, float mean = 0f, float std = 1f)
        {
            var tensor = new CuTensor(shape);
            var values = new float[tensor.Size];

            for (var i = 0; i < values.Length; i++)
            {
                values[i] = NextNormal(mean, std);
            }

            tensor.CopyToDevice(values);
            return tensor;
        }

        private static float NextNormal(float mean = 0f, float std = 1f)
        {
            var u1 = 1f - System.Random.Shared.NextSingle();
            var u2 = 1f - System.Random.Shared.NextSingle();
            var magnitude = std * MathF.Sqrt(-2f * MathF.Log(u1));
            var normal = magnitude * MathF.Sin(2f * MathF.PI * u2) + mean;
            return  normal;
        }
    }
}
