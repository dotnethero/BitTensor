using BitTensor.Abstractions;

namespace BitTensor.Core;

public partial class Tensor
{
    public static Tensor Zero { get; } = Create(0);

    public static Tensor One { get; } = Create(1);
    
    public static Tensor Allocate(int[] shape) =>
        new(shape);

    public static Tensor FromArray(int[] shape, float[] values) => 
        new(shape, new HostAllocation(values));

    public static Tensor Create(float value) =>
        FromArray(shape: [], [value]);

    public static Tensor Create(float[] values) =>
        FromArray(shape: [values.Length], values);

    public static Tensor Create(float[][] values) =>
        FromArray(shape: [values.Length, values[0].Length], values.Collect2D());

    public static Tensor Create(float[][][] values) =>
        FromArray(shape: [values.Length, values[0].Length, values[0][0].Length], values.Collect3D());

    public static Tensor Ones(int[] shape) => Fill(shape, 1f);

    public static Tensor Zeros(int[] shape) => Fill(shape, 0f);

    public static Tensor Fill(int[] shape, float value)
    {
        var values = new float[shape.Product()];
        Array.Fill(values, value);
        return FromArray(shape, values);
    }
    
    public static Tensor Arrange(float start, float end, float step = 1f)
    {
        var count = (int) MathF.Floor((end - start) / step);
        var values = new float[count];
        var index = 0;
        for (var i = start; i < end; i += step)
        {
            values[index++] = i;
        }
        return Create(values);
    }
}
