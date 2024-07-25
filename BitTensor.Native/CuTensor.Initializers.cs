using BitTensor.Abstractions;

namespace BitTensor.CUDA;

public partial class CuTensor // Initializers
{
    public static CuTensor Zero => Create(0);

    public static CuTensor One => Create(1);

    public static CuTensor Create(float value) => 
        new(shape: [], values: [value]);
    
    public static CuTensor Create(float[] values) =>
        new(shape: [values.Length], values);

    public static CuTensor Create(float[][] values) =>
        new(shape: [values.Length, values[0].Length], values.Collect2D());

    public static CuTensor Create(float[][][] values) =>
        new(shape: [values.Length, values[0].Length, values[0][0].Length], values.Collect3D());
}
