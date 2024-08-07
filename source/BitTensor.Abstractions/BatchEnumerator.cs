namespace BitTensor.Abstractions;

public readonly record struct Batch<T>(
    T A,
    T B,
    T Result,
    int BatchIndexA = 0,
    int BatchIndexB = 0,
    int BatchIndexR = 0) 
    where T : AbstractTensor;

public readonly unsafe struct BatchEnumerator(int batchCount, int[] aStrides, int[] bStrides, int[] rStrides)
{
    public readonly int BatchCount = batchCount;
    public readonly int Dimensions = rStrides.Length;

    internal readonly bool StridesAreEqual = Strides.AreEqual(aStrides, bStrides);
    
    public IEnumerable<Batch<T>> GetBatches<T>(T a, T b, T r) where T : AbstractTensor
    {
        var iterator = new Batch<T>(a, b, r);
        for (var batchIndex = 0; batchIndex < BatchCount; batchIndex++)
        {
            var (aIndex, bIndex) = ConvertIndex(batchIndex);
            yield return iterator with
            {
                BatchIndexA = aIndex,
                BatchIndexB = bIndex,
                BatchIndexR = batchIndex,
            };
        }
    }

    private (int aIndex, int bIndex) ConvertIndex(int batchIndex)
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