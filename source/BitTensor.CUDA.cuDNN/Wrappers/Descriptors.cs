﻿using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal static unsafe class Descriptors
{
    public static cudnnBackendDescriptor_t* Create(cudnnBackendDescriptorType_t type)
    {
        cudnnBackendDescriptor_t* descriptor;
        cudnnStatus_t status = cuDNN.cudnnBackendCreateDescriptor(type, &descriptor);
        Status.EnsureIsSuccess(status);
        return descriptor;
    }
    
    public static cudnnBackendDescriptor_t* Finalize(cudnnBackendDescriptor_t* descriptor)
    {
        var status = cuDNN.cudnnBackendFinalize(descriptor);
        Status.EnsureIsSuccess(status);
        return descriptor;
    }
}