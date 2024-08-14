using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public static class CuDebug
{
    public static void WriteLine<T>(CudaNode<T> node, [CallerArgumentExpression(nameof(node))] string tensorName = "") where T : unmanaged, IFloatingPoint<T>
    {
        node.EnsureHasUpdatedValues();
        var tensor = node.Tensor;
        var text = View(tensor);
        Console.WriteLine($"{tensorName}{tensor.Shape} =\n{text}");
    }

    public static void WriteLine<T>(CudaTensor<T> tensor, [CallerArgumentExpression(nameof(tensor))] string tensorName = "") where T : unmanaged, IFloatingPoint<T>
    {
        var text = View(tensor);
        Console.WriteLine($"{tensorName}{tensor.Shape} =\n{text}");
    }

    public static string View<T>(CudaNode<T> node, int dimensionsPerLine = 1) where T : unmanaged, IFloatingPoint<T>
    {
        var text = View(node.Tensor);
        return text;
    }

    public static string View<T>(CudaTensor<T> tensor, int dimensionsPerLine = 1) where T : unmanaged, IFloatingPoint<T>
    {
        IDeviceArray<T> array = tensor;
        var values = array.CopyToHost();
        var shape = tensor.Shape;
        return View(values, shape, dimensionsPerLine);
    }

    public static string View<T>(T[] values, Shape shape, int dimensionsPerLine = 1) where T : IFloatingPoint<T>
    {
        if (values.Length == 0)
        {
            return "[]";
        }

        var dimensions = shape.Dimensions;
        if (dimensions == 0)
        {
            return values[0].ToString("0.00#", CultureInfo.InvariantCulture);
        }

        var sb = new StringBuilder();
        var products = new List<int>();
        var product = 1;
        for (var i = dimensions - 1; i >= 0; --i)
        {
            product *= shape[i];
            products.Add(product);
        }

        var s1 = Shape.GetStrides(shape.Extents);
        var s2 = shape.Strides;

        for (var i = 0; i < values.Length; ++i)
        {
            var leftover = i;
            var translate = 0;
            for (var k = 0; k < dimensions; ++k)
            {
                var di = leftover / s1[k]; // dimension index
                translate += s2[k] * di;
                leftover -= di * s1[k];
            }

            var opens = products.Count(p => (i) % p == 0);
            var closes = products.Count(p => (i + 1) % p == 0);
            var value = values[translate].ToString("0.00#", CultureInfo.InvariantCulture).PadLeft(dimensions > 1 ? 6 : 0);

            if (opens > 0)
                sb.Append(new string(' ', dimensions - opens));

            sb.Append(new string('[', opens));

            if (opens > 0)
                sb.Append(" ");

            sb.Append($"{value} ");
            sb.Append(new string(']', closes));

            if (closes >= dimensionsPerLine && i != values.Length - 1)
                sb.AppendLine();
        }

        return sb.ToString();
    }
}