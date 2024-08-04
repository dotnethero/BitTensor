using BitTensor.Abstractions;

namespace BitTensor.CUDA.ComputeOnly;

internal static class Broadcast
{
    /// <summary>
    /// Verifies if shape can be broadcasted to another shape
    /// </summary>
    public static bool IsBroadcastSupported(int[] source, int[] destination)
    {
        var sd = source.Length;
        var dd = destination.Length;
        if (dd < sd)
            return false;

        for (var i = 0; i < dd; ++i)
        {
            var se = i < sd ? source[^(i+1)] : 1;
            var de = destination[^(i+1)];
            if (se != de && se != 1)
                return false;
        }

        return true;
    }

    public static void EnsureBroadcastIsSupported(int[] source, int[] destination)
    {
        if (!IsBroadcastSupported(source, destination))
            throw new InvalidOperationException($"Broadcast from {source.Serialize()} to {destination.Serialize()} is not supported");
    }
}