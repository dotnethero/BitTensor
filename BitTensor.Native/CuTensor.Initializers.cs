using BitTensor.Abstractions;

namespace BitTensor.CUDA;

public partial class CuTensor // Initializers
{
    public static CuTensor Zero => Create(0);

    public static CuTensor One => Create(1);
    
    public static CuTensor Ones(int[] shape) => Fill(shape, 1f);

    public static CuTensor Zeros(int[] shape) => Fill(shape, 0f);

    public static CuTensor Fill(int[] shape, float value)
    {
        var values = new float[shape.Product()];
        Array.Fill(values, value); // TODO: use CUDA
        return new(shape, values);
    }

    public static CuTensor Create(float value) => 
        new(shape: [], values: [value]);
    
    public static CuTensor Create(float[] values) =>
        new(shape: [values.Length], values);

    public static CuTensor Create(float[][] values) =>
        new(shape: [values.Length, values[0].Length], values.Collect2D());

    public static CuTensor Create(float[][][] values) =>
        new(shape: [values.Length, values[0].Length, values[0][0].Length], values.Collect3D());
}
