using BitTensor.Abstractions;
using BitTensor.CUDA.ComputeOnly.Wrappers;
using BitTensor.CUDA.Interop;

// ReSharper disable NotAccessedVariable
// ReSharper disable JoinDeclarationAndInitializer

namespace BitTensor.CUDA.ComputeOnly;

using static cuBLAS;
using static cuTENSOR;

public partial class CuTensor
{
    public static CuTensor operator +(CuTensor a, CuTensor b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        Add(a, b, output);
        return output;
    }

    public static CuTensor operator *(CuTensor a, CuTensor b)
    {
        var batchDimensions = Shapes.EnsureShapesAreCompatible(a.Shape[..^2], b.Shape[..^2]);
        var output = new CuTensor([..batchDimensions, a.PrevDimension, b.LastDimension]);
        Multiply(a, b, output);
        return output;
    }

    // inplace operations

    public static void Add(CuTensor a, CuTensor b, CuTensor c)
    {
        using var context = new CuTensorContext();

        using var a1 = context.CreateDescriptor(a);
        using var b1 = context.CreateDescriptor(b);
        using var c1 = context.CreateDescriptor(c);

        using var operation = context.CreateElementwiseAdd(a1, b1, c1);

        operation.Execute();
    }

    public static unsafe void Multiply(CuTensor a, CuTensor b, CuTensor c)
    {
        var context = new CublasContext();

        context.Gemm(a, b, c);
    }
}