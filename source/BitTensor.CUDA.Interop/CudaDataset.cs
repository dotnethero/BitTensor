using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA;

public unsafe class CudaDataset<T> : IDisposable where T : unmanaged
{
    public readonly Shape Shape;
    public readonly T[] Data;
    public readonly T* Pointer;
    public readonly int Size;
    public readonly int ElementSize;

    public CudaDataset(Shape shape, T[] data)
    {
        Shape = shape;
        Data = data;
        Size = data.Length;
        ElementSize = sizeof(T);

        fixed (T* pointer = Data)
        {
            Pointer = pointer;
            var status = cudaRT.cudaHostRegister(pointer, (uint)(Size * ElementSize), 0x00);
            if (status != cudaError.cudaSuccess)
                throw new InvalidOperationException("Failed to page lock host memory");
        }
    }
    
    public void Dispose()
    {
        cudaRT.cudaHostUnregister(Pointer);
    }
}