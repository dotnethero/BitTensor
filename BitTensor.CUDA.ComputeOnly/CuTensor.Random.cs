namespace BitTensor.CUDA.ComputeOnly;

public partial class CuTensor
{
    public static class Random // TODO: use cuRAND
    {
        public static CuTensor Uniform(int[] shape, float min = -1f, float max = +1f)
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
            return System.Random.Shared.NextSingle() * (max - min) + min;
        }
    }
}
