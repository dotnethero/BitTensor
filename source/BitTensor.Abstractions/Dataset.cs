namespace BitTensor.Abstractions;

public record Dataset<T>(Shape Shape, T[] Data);
