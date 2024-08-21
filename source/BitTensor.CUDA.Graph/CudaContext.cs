using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph;

public class CudaContext : IDisposable
{
    internal readonly List<IDeviceArray> DeviceArrays = [];

    public CuRandContext cuRAND { get; }
    public CuTensorContext cuTENSOR { get; }

    public static CudaContext CreateDefault()
    {
        return new CudaContext();
    }
    
    public static CudaContext GetContext<T>(
        CudaNode<T> operand) 
        where T : unmanaged, IFloatingPoint<T> =>
        operand.Context;

    public static CudaContext GetContext<T>(
        params CudaNode<T>[] operands)
        where T : unmanaged, IFloatingPoint<T> =>
        operands 
            .Select(c => c.Context)
            .Distinct()
            .Single();

    private CudaContext()
    {
        cuRAND = new CuRandContext();
        cuTENSOR = new CuTensorContext();
    }

    public void Synchronize()
    {
        cudaRT.cudaDeviceSynchronize();
    }
    
    public CudaTensor<T> Allocate<T>(Shape shape) 
        where T : unmanaged => 
        AddArray(new CudaTensor<T>(shape));

    public CudaTensor<T> Allocate<T>(Shape shape, T[] values) 
        where T : unmanaged => 
        AddArray(new CudaTensor<T>(shape, values));

    private TArray AddArray<TArray>(TArray array) where TArray : IDeviceArray
    {
        DeviceArrays.Add(array);
        return array;
    }
    
    public void Dispose()
    {
        cuTENSOR.Dispose();
        FreeArrays();
    }

    private void FreeArrays()
    {
        var bytes = 0;
        var arrays = 0;
        foreach (var array in DeviceArrays)
        {
            array.Dispose();
            bytes += array.Size + array.ElementSize;
            arrays++;
        }
        Console.Error.WriteLine($"{arrays} arrays ({bytes >> 10} kB) disposed");
    }
}