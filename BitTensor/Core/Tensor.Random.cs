namespace BitTensor.Core;

public partial class Tensor
{
    public static class Random
    {
        public static Tensor Uniform(int[] shape, float min = -1f, float max = +1f)
        {
            var values = new float[shape.Product()];
            for (var i = 0; i < values.Length; i++)
            {
                values[i] = NextUniform(min, max);
            }

            return new(shape, values);
        }
        
        private static float NextUniform(float min = -1f, float max = 1f)
        {
            return System.Random.Shared.NextSingle() * (max - min) + min;
        }

        public static Tensor Normal(int[] shape, float mean = 0f, float std = 1f)
        {
            var values = new float[shape.Product()];
            for (var i = 0; i < values.Length; i++)
            {
                values[i] = NextNormal(mean, std);
            }

            return new(shape, values);
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
