namespace BitTensor.Abstractions;

public readonly unsafe struct BatchStrides(int batchCount, int[] aStrides, int[] bStrides, int[] rStrides)
{
    public readonly int BatchCount = batchCount;
    public readonly int Dimensions = rStrides.Length;

    internal readonly bool StridesAreEqual =
        Shapes.AreEqual(aStrides, bStrides);

    public (int aIndex, int bIndex) ConvertIndex(int batchIndex)
    {
        if (StridesAreEqual)
            return (batchIndex, batchIndex);

        var aIndex = 0;
        var bIndex = 0;
        var leftover = batchIndex;
        fixed (int* ap = aStrides, bp = bStrides, rp = rStrides)
        {
            for (var i = 0; i < Dimensions; ++i)
            {
                var di = leftover / rp[i]; // dimension index
                aIndex += ap[i] * di;
                bIndex += bp[i] * di;
                leftover -= di * rp[i];
            }
        }
        return (aIndex, bIndex);
    }
}

public static class Batching
{
    public static BatchStrides GetBatchStrides(AbstractTensor a, AbstractTensor b, Range dimensions)
    {
        var batchDims = Math.Max(a.Dimensions, b.Dimensions) - 2;

        var aBatchShapeOrig = a.Shape[dimensions];
        var bBatchShapeOrig = b.Shape[dimensions];

        var aBatchShape = new int[batchDims];
        var bBatchShape = new int[batchDims];
        var rBatchShape = new int[batchDims];

        for (var i = 0; i < batchDims; ++i)
        {
            var ai = i >= aBatchShapeOrig.Length ? 1 : aBatchShapeOrig[^(i+1)];
            var bi = i >= bBatchShapeOrig.Length ? 1 : bBatchShapeOrig[^(i+1)];
            aBatchShape[^(i+1)] = ai;
            bBatchShape[^(i+1)] = bi;
            rBatchShape[^(i+1)] = ai >= bi ? ai : bi;
        }

        var aStrides = aBatchShape.GetStrides();
        var bStrides = bBatchShape.GetStrides();
        var rStrides = rBatchShape.GetStrides();
        
        for (var i = 0; i < batchDims; ++i)
        {
            if (aBatchShape[i] == 1)
                aStrides[i] = 0;

            if (bBatchShape[i] == 1)
                bStrides[i] = 0;
        }

        var batchCount = batchDims == 0 ? 1 : rStrides[0] * rBatchShape[0];
        return new(batchCount, aStrides, bStrides, rStrides);
    }
}