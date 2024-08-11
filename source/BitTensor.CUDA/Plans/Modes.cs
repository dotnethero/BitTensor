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
            .Where((_, i) => !axis.Contains(i))
            .ToArray();
    }
}