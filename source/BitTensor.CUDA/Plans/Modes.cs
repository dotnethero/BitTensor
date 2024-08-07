﻿using BitTensor.Abstractions;

namespace BitTensor.CUDA.Plans;

internal static class Modes
{
    public static int[] GetOrdinaryModes(this Shape shape, int offset = 0, int? skip = null)
    {
        var dims = shape.Dimensions;
        if (dims == 0)
            return [];

        var modes = new int[dims];

        for (var i = 1; i <= dims; ++i)
        {
            modes[^i] = i >= skip
                ? i + offset + 1
                : i + offset;
        }

        return modes;
    }

    public static int[] GetReductionModes(this Shape shape, HashSet<int> axis)
    {
        return shape
            .GetOrdinaryModes()
            .Where((s, i) => !axis.Contains(i))
            .ToArray();
    }

    public static (int[] leftModes, int[] rightModes, int[] resultModes) GetMultiplicationModes(Shape left, Shape right, Shape result)
    {
        var leftModes = left.GetOrdinaryModes();
        var rightModes = right.GetOrdinaryModes();
        var resultModes = result.GetOrdinaryModes();

        var contractionMode = Math.Max(left.Dimensions, right.Dimensions) + 1;

        if (left.Dimensions == 0 ||
            right.Dimensions == 0) 
            return (leftModes, rightModes, resultModes);

        if (right.Dimensions == 1)
        {
            leftModes[^1] = contractionMode;
            rightModes[^1] = contractionMode;
            resultModes = result.GetOrdinaryModes(offset: +1);
        }
        else if (left.Dimensions == 1)
        {
            var contractedMode = rightModes[^2];
            leftModes[^1] = contractionMode;
            rightModes[^2] = contractionMode;
            resultModes = result.GetOrdinaryModes(skip: contractedMode);
        }
        else
        {
            leftModes[^1] = contractionMode;
            rightModes[^2] = contractionMode;
            resultModes = result.GetOrdinaryModes();
        }

        return (leftModes, rightModes, resultModes);
    }
}