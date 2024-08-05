﻿using System.Runtime.CompilerServices;
using System.Text;

namespace BitTensor.CUDA.ComputeOnly;

public static class CuDebug
{
    public static void WriteLine(CuTensor tensor, [CallerArgumentExpression("tensor")] string tensorName = "")
    {
        var values = tensor.CopyToHost();
        var shape = tensor.Shape;
        var text = View(values, shape, krStyle: true);
        Console.WriteLine($"{tensorName} = {text}");
    }

    public static string View(float[] values, int[] shape, int dimensionsPerLine = 1, bool krStyle = false)
    {
        if (values.Length == 0)
        {
            return "[]";
        }

        var dimensions = shape.Length;
        if (dimensions == 0)
        {
            return values[0].ToString("0.00#");
        }

        var sb = new StringBuilder();
        var products = new List<int>();
        var product = 1;
        for (var i = dimensions - 1; i >= 0; --i)
        {
            product *= shape[i];
            products.Add(product);
        }

        for (var i = 0; i < values.Length; ++i)
        {
            var opens = products.Count(p => (i) % p == 0);
            var closes = products.Count(p => (i + 1) % p == 0);
            var value = values[i].ToString("0.00#").PadLeft(dimensions > 1 ? 6 : 0);

            if (opens > 0)
                sb.Append(new string(' ', dimensions - opens));

            if (krStyle && opens > 1)
            {
                if (i == 0)
                {
                    sb.AppendLine("[");
                    sb.Append(new string(' ', dimensions - opens + 1));
                }

                sb.Append(new string('[', opens - 1));
            }
            else
            {
                sb.Append(new string('[', opens));
            }

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