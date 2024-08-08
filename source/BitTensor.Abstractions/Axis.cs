namespace BitTensor.Abstractions;

public static class Axis
{
    public static int[] GetOrdinaryAxis(this Shape shape)
    {
        var dims = shape.Dimensions;
        var axis = new int[dims];

        for (var i = 0; i < dims; ++i)
        {
            axis[i] = i;
        }

        return axis;
    }

    public static int[] GetTransposeAxis(this Shape shape)
    {
        var axis = GetOrdinaryAxis(shape);
        var dims = shape.Dimensions;
        if (dims < 2)
            return axis;

        (axis[^1], axis[^2]) = (axis[^2], axis[^1]);

        return axis;
    }
}