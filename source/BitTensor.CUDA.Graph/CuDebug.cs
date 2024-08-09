using System.Runtime.CompilerServices;

namespace BitTensor.CUDA.Graph;

public static class CuGraphDebug
{
    public static void WriteLine(CuTensorNode node, [CallerArgumentExpression("node")] string tensorName = "")
    {
        var text = View(node);
        Console.WriteLine($"{tensorName} =\n{text}");
    }

    public static string View(CuTensorNode node, int dimensionsPerLine = 1)
    {
        node.EnsureHasUpdatedValues();
        return CuDebug.View(node.Tensor);
    }
}