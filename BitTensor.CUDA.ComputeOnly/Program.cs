using BitTensor.CUDA.ComputeOnly.Graph;

namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        using var a = CuTensor.Random.Uniform([3, 4]);
        using var b = CuTensor.Random.Uniform([3, 1]);
        using var c = CuTensor.Random.Uniform([1, 4]);

        using var na = new CuTensorNode(a);
        using var nb = new CuTensorNode(b);
        using var nc = new CuTensorNode(c);
        using var nx = na + nb;
        using var ny = na + nc;

        CuDebug.WriteLine(na);
        CuDebug.WriteLine(nb);
        CuDebug.WriteLine(nc);
        CuDebug.WriteLine(nx);
        CuDebug.WriteLine(ny);
        
        var nxgrads = nx.GetGradients();
        CuDebug.WriteLine(nxgrads[na]);
        CuDebug.WriteLine(nxgrads[nb]);

        var nygrads = ny.GetGradients();
        CuDebug.WriteLine(nygrads[na]);
        CuDebug.WriteLine(nygrads[nc]);
    }
}