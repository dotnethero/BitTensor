namespace BitTensor.Abstractions;

public static class Batching
{
    public static BatchEnumerator GetBatchEnumerator(AbstractTensor a, AbstractTensor b, Range dimensions)
    {
        var batchDims = Math.Max(a.Dimensions, b.Dimensions) - 2;

        var aBatchShapeOrig = a.Shape[dimensions];
        var bBatchShapeOrig = b.Shape[dimensions];

        var aBatchShape = new int[batchDims];
        var bBatchShape = new int[batchDims];
        var rBatchShape = new int[batchDims];

        for (var i = 0; i < batchDims; ++i)
        {
            var ai = i >= aBatchShapeOrig.Dimensions ? 1 : aBatchShapeOrig[^(i+1)];
            var bi = i >= bBatchShapeOrig.Dimensions ? 1 : bBatchShapeOrig[^(i+1)];
            aBatchShape[^(i+1)] = ai;
            bBatchShape[^(i+1)] = bi;
            rBatchShape[^(i+1)] = ai >= bi ? ai : bi;
        }

        var aStrides = Shape.GetStrides(aBatchShape);
        var bStrides = Shape.GetStrides(bBatchShape);
        var rStrides = Shape.GetStrides(rBatchShape);
        
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