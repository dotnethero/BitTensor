using BitTensor.CUDA.ComputeOnly.Graph;

namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        using var a = CuTensor.Random.Uniform([2, 3, 4]);
        using var b = CuTensor.Random.Uniform([3, 1]);
        using var c = CuTensor.Random.Uniform([1, 4]);
        using var d = CuTensor.Random.Uniform([4, 5]);

        using var z1 = CuTensor.Allocate([2, 3, 5]);
        using var z2 = CuTensor.Allocate([2, 3, 5]);

        CuTensor.Multiply(a, d, z1);
        CuBLAS.Multiply(a, d, z2);

        CuDebug.WriteLine(a);
        CuDebug.WriteLine(z1);
        CuDebug.WriteLine(z2);

        return;

        using var na = new CuTensorNode(a);
        using var nb = new CuTensorNode(b);
        using var nc = new CuTensorNode(c);
        using var nx = na + nb; // TODO: responsible for dispose
        using var ny = na + nc;

        CuDebug.WriteLine(na);
        CuDebug.WriteLine(nb);
        CuDebug.WriteLine(nc);
        CuDebug.WriteLine(nx);
        CuDebug.WriteLine(ny);
        
        using var nxgrads = nx.GetGradients();
        CuDebug.WriteLine(nxgrads[na]);
        CuDebug.WriteLine(nxgrads[nb]);

        using var nygrads = ny.GetGradients();
        CuDebug.WriteLine(nygrads[na]);
        CuDebug.WriteLine(nygrads[nc]);
    }
}