using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph;

public partial class CudaContext : IDisposable
{
    internal readonly List<IDeviceArray> DeviceArrays = [];
    internal readonly List<IDisposable> Resources = [];

    public CuRandContext cuRAND { get; }
    public CuTensorContext cuTENSOR { get; }

    public static CudaContext CreateDefault()
    {
        return new CudaContext();
    }

    private CudaContext()
    {
        cuRAND = new CuRandContext();
        cuTENSOR = new CuTensorContext();
    }

    public void Synchronize()
    {
        cudaRT.cudaDeviceSynchronize();
    }
    
    public CudaNode<T> CreateNode<T>(Shape shape) 
        where T : unmanaged, IFloatingPoint<T> =>
        new(this, Allocate<T>(shape));

    public CudaNode<T> CreateNode<T>(Shape shape, T[] values)
        where T : unmanaged, IFloatingPoint<T> =>
        new(this, Allocate<T>(shape, values));

    public CudaNode<T> CreateNode<T>(T value)
        where T : unmanaged, IFloatingPoint<T> =>
        new(this, Allocate<T>([], [value]));

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
    
    private TResouce AddResource<TResouce>(TResouce resouce) where TResouce : IDisposable
    {
        Resources.Add(resouce);
        return resouce;
    }

    public void Dispose()
    {
        cuTENSOR.Dispose();
        FreeResources();
        FreeArrays();
    }

    private void FreeResources()
    {
        var plans = 0;
        foreach (var resource in Resources)
        {
            if (resource is ICuTensorPlan)
                plans++;

            resource.Dispose();
        }
        Console.WriteLine($"{plans} operation plans disposed");
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
        Console.WriteLine($"{arrays} arrays ({bytes >> 10} kB) disposed");
    }
}