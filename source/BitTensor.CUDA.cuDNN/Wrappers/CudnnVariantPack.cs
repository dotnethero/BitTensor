using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;
using DescriptorType = cudnnBackendDescriptorType_t;

public sealed unsafe class CudnnVariantPack<T> : IDisposable where T : unmanaged, IFloatingPoint<T>
{
    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnVariantPack(IUniqueDeviceArray<T>[] tensors)
    {
        var count = tensors.Length;
        var uniques = stackalloc long[count];
        var pointers = stackalloc void*[count];

        for (var i = 0; i < count; i++)
        {
            uniques[i] = tensors[i].UniqueId;
            pointers[i] = tensors[i].Pointer;
        }

        void* workspace;

        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);

        SetPointers(descriptor, count, pointers);
        SetUniqueIds(descriptor, count, uniques);
        SetWorkspace(descriptor, &workspace);

        Descriptor = Descriptors.Finalize(descriptor);
    }
    
    private static void SetPointers(cudnnBackendDescriptor_t* descriptor, int count, void** pointers)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
            AttributeType.CUDNN_TYPE_VOID_PTR, 
            count,
            pointers);

        Status.EnsureIsSuccess(status);
    }
    
    private static void SetUniqueIds(cudnnBackendDescriptor_t* descriptor, int count, long* uniques)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
            AttributeType.CUDNN_TYPE_INT64, 
            count,
            uniques);

        Status.EnsureIsSuccess(status);
    }
    
    public static void SetWorkspace(cudnnBackendDescriptor_t* descriptor, void** workspace)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
            AttributeType.CUDNN_TYPE_VOID_PTR,
            elementCount: 1,
            workspace);

        Status.EnsureIsSuccess(status);
    }

    public void Dispose()
    {
        cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
    }
}