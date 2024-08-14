namespace BitTensor.Abstractions;

public static class Axis
{
    public static Index[] InvertPermutation(Index[] axis)
    {
        var dims = axis.Length;
        var inverted = new Index[dims];

        for (var i = 0; i < dims; i++)
        {
            inverted[axis[i]] = i;
        }

        return inverted;
    }

    public static Index[] GetOrdinaryAxis(this Shape shape)
    {
        var dims = shape.Dimensions;
        var axis = new Index[dims];

        for (var i = 0; i < dims; ++i)
        {
            axis[i] = i;
        }

        return axis;
    }

    public static Index[] GetTransposeAxis(this Shape shape)
    {
        var axis = GetOrdinaryAxis(shape);
        var dims = shape.Dimensions;
        if (dims < 2)
            return axis;

        (axis[^1], axis[^2]) = (axis[^2], axis[^1]);

        return axis;
    }

    public static bool AxisAreUnique(this Shape shape, Index[] items) =>
        items.Select(shape.GetOffset).ToHashSet().Count == items.Length;
    
    public static string ToText(this Index[] items) => 
        $"({string.Join(",", items)})";
}