using BitTensor.Abstractions;
using ILGPU.Runtime;

namespace BitTensor.CUDA;

public class CuAllocator(Accelerator accelerator) : ITensorAllocator<CuTensor>
{
    public CuTensor Allocate(int[] shape) => 
        new(accelerator, shape: shape);

    public CuTensor Create(float value) => 
        new(accelerator, shape: [], values: [value]);
    
    public CuTensor Create(float[] values) =>
        new(accelerator, shape: [values.Length], values);

    public CuTensor Create(float[][] values) =>
        new(accelerator, shape: [values.Length, values[0].Length], values.Collect2D());

    public CuTensor Create(float[][][] values) =>
        new(accelerator, shape: [values.Length, values[0].Length, values[0][0].Length], values.Collect3D());
}
