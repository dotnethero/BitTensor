namespace BitTensor.Abstractions;

public static class Shapes
{
    public static bool AreEqual(Shape a, Shape b)
    {
        if (a.Dimensions != b.Dimensions)
            return false;

        for (var i = 0; i < a.Dimensions; i++)
        {
            if (a[i] != b[i])
                return false;
        }

        return true;
    }
    
    public static void EnsureAreEqual(Shape a, Shape b)
    {
        if (!AreEqual(a, b))
            throw new InvalidOperationException($"Shapes are not equal: {a} and {b}");
    }

    public static bool TryBroadcast(Shape a, Shape b, out Shape shape)
    {
        var length = Math.Max(a.Dimensions, b.Dimensions);
        var extents = new int[length];

        shape = null!;

        for (var i = 0; i < length; ++i)
        {
            var ai = i >= a.Dimensions ? 1 : a[^(i+1)];
            var bi = i >= b.Dimensions ? 1 : b[^(i+1)];
            if (ai != bi && ai != 1 && bi != 1)
                return false;

            extents[^(i+1)] = Math.Max(ai, bi);
        }

        shape = Shape.Create(extents);

        return true;
    }

    public static Shape Broadcast(Shape a, Shape b)
    {
        if (!TryBroadcast(a, b, out var shape))
            throw new InvalidOperationException($"Shapes are not compatible for broadcast: {a} and {b}");

        return shape;
    }

    public static Shape BroadcastMatMul(Shape a, Shape b)
    {
        if (a.Dimensions == 0)
            return b;

        if (b.Dimensions == 0)
            return a;
        
        if (a.Dimensions == 1)
            return b[1..];

        if (b.Dimensions == 1) 
            return a[..^1];
        
        if (a[^1] != b[^2])
            throw new InvalidOperationException($"Shapes are not compatible for matrix multiplication: {a} and {b}");

        var batches = Broadcast(a[..^2], b[..^2]);

        return [..batches, a[^2], b[^1]];
    }

    public static HashSet<int> GetBroadcastedAxis(Shape input, Shape output)
    {
        var id = input.Dimensions;
        var od = output.Dimensions;
        var axis = new HashSet<int>(od);

        for (var i = 1; i <= od; ++i)
        {
            if (i > id || input[^i] != output[^i])
                axis.Add(od - i);
        }

        return axis;
    }
}
