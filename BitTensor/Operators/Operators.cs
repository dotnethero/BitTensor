namespace BitTensor.Operators;

internal interface IBinaryOperator<T>
{
    static abstract T Execute(T a, T b);
    static abstract void Execute(ReadOnlySpan<T> a, T b, Span<T> result);
    static abstract void Execute(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> result);
}

internal interface IAggregateOperator<T> : IBinaryOperator<T>
{
    static abstract T Identity { get; }
    static abstract T Aggregate(ReadOnlySpan<T> a);
}
