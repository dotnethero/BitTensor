namespace BitTensor.Abstractions;

public abstract class AbstractTensor
{
    internal static long MaxID;

    public readonly long Id;
    public readonly int Size;
    public readonly int Dimensions;
    public readonly int[] Shape;
    public readonly int[] Strides;
    
    // tensor properties
    public readonly bool IsEmpty;
    public readonly bool IsScalar;
    public readonly bool IsVector;
    public readonly bool IsRow;
    public readonly bool IsColumn;

    // helpers
    public readonly int PrevDimension;
    public readonly int LastDimension;

    protected internal AbstractTensor(int[] shape)
    {
        var size = shape.Product();

        Id = Interlocked.Increment(ref MaxID);
        Size = size;
        Shape = shape;
        Strides = shape.GetStrides();
        Dimensions = shape.Length;

        IsEmpty = Size == 0;
        IsScalar = Dimensions == 0;
        IsVector = Dimensions == 1;
        
        if (Dimensions > 0)
        {
            LastDimension = Shape[Dimensions - 1];
            IsRow = LastDimension == Size;
        }

        if (Dimensions > 1)
        {
            PrevDimension = Shape[Dimensions - 2];
            IsColumn = PrevDimension == Size;
        }
    }
}