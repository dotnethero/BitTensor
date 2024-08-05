namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        using var a = CuTensor.Random.Uniform([3, 4]);
        using var b = CuTensor.Random.Uniform([3, 1]);
        using var c = CuTensor.Random.Uniform([1, 4]);

        using var inputA = new CuTensorNode(a);
        using var inputB = new CuTensorNode(b);
        using var output = inputA + inputB;

        output.EnsureHasUpdatedValues();

        CuDebug.WriteLine(inputA.Tensor);
        CuDebug.WriteLine(inputB.Tensor);
        CuDebug.WriteLine(output.Tensor);
    }
}