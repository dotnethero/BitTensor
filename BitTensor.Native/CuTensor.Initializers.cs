using BitTensor.Abstractions;
using ILGPU.Runtime;

namespace BitTensor.CUDA;

public partial class CuTensor // Initializers
{
    public static CuTensor Zero => Create(null, 0);

    public static CuTensor One => Create(null, 1);
    
    public static CuTensor Ones(int[] shape) => Fill(null, shape, 1f);

    public static CuTensor Zeros(int[] shape) => Fill(null, shape, 0f);

    public static CuTensor Fill(Accelerator accelerator, int[] shape, float value)
    {
        var values = new float[shape.Product()];
        Array.Fill(values, value); // TODO: use CUDA
        return new(accelerator, shape, values);
    }

    public static CuTensor Create(Accelerator accelerator, float value) => 
        new(accelerator, shape: [], values: [value]);
    
    public static CuTensor Create(Accelerator accelerator, float[] values) =>
        new(accelerator, shape: [values.Length], values);

    public static CuTensor Create(Accelerator accelerator, float[][] values) =>
        new(accelerator, shape: [values.Length, values[0].Length], values.Collect2D());

    public static CuTensor Create(Accelerator accelerator, float[][][] values) =>
        new(accelerator, shape: [values.Length, values[0].Length, values[0][0].Length], values.Collect3D());
}
