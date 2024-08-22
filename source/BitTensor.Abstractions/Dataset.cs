using System.Data;

namespace BitTensor.Abstractions;

public record Dataset<T>(Shape Shape, T[] Data)
{
    int[] GetRandomBatchIndexes(int batchSize)
    {
        var indexes = Enumerable.Range(0, batchSize).ToArray();
        Random.Shared.Shuffle(indexes);
        return indexes;
    }
}
