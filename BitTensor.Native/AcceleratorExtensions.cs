using BitTensor.Abstractions;
using ILGPU.Runtime;

namespace BitTensor.CUDA;

public static class AcceleratorExtensions
{
    public static CuTensor AllocateZero(Accelerator accelerator) => new(accelerator, [], [0]);

    public static CuTensor AllocateOne(Accelerator accelerator)  => new(accelerator, [], [1]);

    public static CuTensor CreateTensor(this Accelerator accelerator, float value) => 
        new(accelerator, shape: [], values: [value]);
    
    public static CuTensor CreateTensor(this Accelerator accelerator, float[] values) =>
        new(accelerator, shape: [values.Length], values);

    public static CuTensor CreateTensor(this Accelerator accelerator, float[][] values) =>
        new(accelerator, shape: [values.Length, values[0].Length], values.Collect2D());

    public static CuTensor CreateTensor(this Accelerator accelerator, float[][][] values) =>
        new(accelerator, shape: [values.Length, values[0].Length, values[0][0].Length], values.Collect3D());
}
