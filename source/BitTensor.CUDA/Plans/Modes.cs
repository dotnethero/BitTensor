using BitTensor.Abstractions;

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
        if (left.Dimensions < 2 ||
            right.Dimensions < 2)
            throw new InvalidOperationException("Can't execute matrix multiplication on vectors and scalars - use dimension padding");

        var leftModes = left.GetOrdinaryModes();
        var rightModes = right.GetOrdinaryModes();
        var resultModes = result.GetOrdinaryModes();

        var contractionMode = Math.Max(left.Dimensions, right.Dimensions) + 1;

        leftModes[^1] = contractionMode;
        rightModes[^2] = contractionMode;

        return (leftModes, rightModes, resultModes);
    }
}