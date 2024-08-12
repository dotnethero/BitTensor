﻿// <auto-generated>
// This code is generated by csbindgen.
// DON'T CHANGE THIS DIRECTLY.
// </auto-generated>
#pragma warning disable CS8500
#pragma warning disable CS8981

using System;
using System.Runtime.InteropServices;

namespace BitTensor.CUDA.Interop;

public static unsafe partial class cuTENSOR
{
    public static readonly cutensorComputeDescriptor* CUTENSOR_COMPUTE_DESC_16F;
    public static readonly cutensorComputeDescriptor* CUTENSOR_COMPUTE_DESC_32F;
    public static readonly cutensorComputeDescriptor* CUTENSOR_COMPUTE_DESC_64F;

    const string __DllName = "cutensor.dll";

    static cuTENSOR()
    {
        var lib = NativeLibrary.Load(__DllName);

        CUTENSOR_COMPUTE_DESC_16F = (cutensorComputeDescriptor*) Helpers.ReadConstant(lib, nameof(CUTENSOR_COMPUTE_DESC_16F));
        CUTENSOR_COMPUTE_DESC_32F = (cutensorComputeDescriptor*) Helpers.ReadConstant(lib, nameof(CUTENSOR_COMPUTE_DESC_32F));
        CUTENSOR_COMPUTE_DESC_64F = (cutensorComputeDescriptor*) Helpers.ReadConstant(lib, nameof(CUTENSOR_COMPUTE_DESC_64F));

        NativeLibrary.Free(lib);
    }

    [DllImport(__DllName, EntryPoint = "cutensorCreate", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorCreate(cutensorHandle** handle);

    [DllImport(__DllName, EntryPoint = "cutensorDestroy", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorDestroy(cutensorHandle* handle);

    [DllImport(__DllName, EntryPoint = "cutensorHandleResizePlanCache", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorHandleResizePlanCache(cutensorHandle* handle, uint numEntries);

    [DllImport(__DllName, EntryPoint = "cutensorHandleWritePlanCacheToFile", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorHandleWritePlanCacheToFile(cutensorHandle* handle, byte* filename);

    [DllImport(__DllName, EntryPoint = "cutensorHandleReadPlanCacheFromFile", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorHandleReadPlanCacheFromFile(cutensorHandle* handle, byte* filename, uint* numCachelinesRead);

    [DllImport(__DllName, EntryPoint = "cutensorWriteKernelCacheToFile", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorWriteKernelCacheToFile(cutensorHandle* handle, byte* filename);

    [DllImport(__DllName, EntryPoint = "cutensorReadKernelCacheFromFile", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorReadKernelCacheFromFile(cutensorHandle* handle, byte* filename);

    [DllImport(__DllName, EntryPoint = "cutensorCreateTensorDescriptor", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorCreateTensorDescriptor(cutensorHandle* handle, cutensorTensorDescriptor** desc, uint numModes, long* extent, long* stride, cutensorDataType_t dataType, uint alignmentRequirement);

    [DllImport(__DllName, EntryPoint = "cutensorDestroyTensorDescriptor", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorDestroyTensorDescriptor(cutensorTensorDescriptor* desc);

    [DllImport(__DllName, EntryPoint = "cutensorCreateElementwiseTrinary", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorCreateElementwiseTrinary(cutensorHandle* handle, cutensorOperationDescriptor** desc, cutensorTensorDescriptor* descA, int* modeA, cutensorOperator_t opA, cutensorTensorDescriptor* descB, int* modeB, cutensorOperator_t opB, cutensorTensorDescriptor* descC, int* modeC, cutensorOperator_t opC, cutensorTensorDescriptor* descD, int* modeD, cutensorOperator_t opAB, cutensorOperator_t opABC, cutensorComputeDescriptor* descCompute);

    [DllImport(__DllName, EntryPoint = "cutensorElementwiseTrinaryExecute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorElementwiseTrinaryExecute(cutensorHandle* handle, cutensorPlan* plan, void* alpha, void* A, void* beta, void* B, void* gamma, void* C, void* D, CUstream_st* stream);

    [DllImport(__DllName, EntryPoint = "cutensorCreateElementwiseBinary", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorCreateElementwiseBinary(cutensorHandle* handle, cutensorOperationDescriptor** desc, cutensorTensorDescriptor* descA, int* modeA, cutensorOperator_t opA, cutensorTensorDescriptor* descC, int* modeC, cutensorOperator_t opC, cutensorTensorDescriptor* descD, int* modeD, cutensorOperator_t opAC, cutensorComputeDescriptor* descCompute);

    [DllImport(__DllName, EntryPoint = "cutensorElementwiseBinaryExecute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorElementwiseBinaryExecute(cutensorHandle* handle, cutensorPlan* plan, void* alpha, void* A, void* gamma, void* C, void* D, CUstream_st* stream);

    [DllImport(__DllName, EntryPoint = "cutensorCreatePermutation", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorCreatePermutation(cutensorHandle* handle, cutensorOperationDescriptor** desc, cutensorTensorDescriptor* descA, int* modeA, cutensorOperator_t opA, cutensorTensorDescriptor* descB, int* modeB, cutensorComputeDescriptor* descCompute);

    [DllImport(__DllName, EntryPoint = "cutensorPermute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorPermute(cutensorHandle* handle, cutensorPlan* plan, void* alpha, void* A, void* B, CUstream_st* stream);

    [DllImport(__DllName, EntryPoint = "cutensorCreateContraction", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorCreateContraction(cutensorHandle* handle, cutensorOperationDescriptor** desc, cutensorTensorDescriptor* descA, int* modeA, cutensorOperator_t opA, cutensorTensorDescriptor* descB, int* modeB, cutensorOperator_t opB, cutensorTensorDescriptor* descC, int* modeC, cutensorOperator_t opC, cutensorTensorDescriptor* descD, int* modeD, cutensorComputeDescriptor* descCompute);

    [DllImport(__DllName, EntryPoint = "cutensorDestroyOperationDescriptor", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorDestroyOperationDescriptor(cutensorOperationDescriptor* desc);

    [DllImport(__DllName, EntryPoint = "cutensorOperationDescriptorSetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorOperationDescriptorSetAttribute(cutensorHandle* handle, cutensorOperationDescriptor* desc, cutensorOperationDescriptorAttribute_t attr, void* buf, nuint sizeInBytes);

    [DllImport(__DllName, EntryPoint = "cutensorOperationDescriptorGetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorOperationDescriptorGetAttribute(cutensorHandle* handle, cutensorOperationDescriptor* desc, cutensorOperationDescriptorAttribute_t attr, void* buf, nuint sizeInBytes);

    [DllImport(__DllName, EntryPoint = "cutensorCreatePlanPreference", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorCreatePlanPreference(cutensorHandle* handle, cutensorPlanPreference** pref, cutensorAlgo_t algo, cutensorJitMode_t jitMode);

    [DllImport(__DllName, EntryPoint = "cutensorDestroyPlanPreference", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorDestroyPlanPreference(cutensorPlanPreference* pref);

    [DllImport(__DllName, EntryPoint = "cutensorPlanPreferenceSetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorPlanPreferenceSetAttribute(cutensorHandle* handle, cutensorPlanPreference* pref, cutensorPlanPreferenceAttribute_t attr, void* buf, nuint sizeInBytes);

    [DllImport(__DllName, EntryPoint = "cutensorPlanGetAttribute", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorPlanGetAttribute(cutensorHandle* handle, cutensorPlan* plan, cutensorPlanAttribute_t attr, void* buf, nuint sizeInBytes);

    [DllImport(__DllName, EntryPoint = "cutensorEstimateWorkspaceSize", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorEstimateWorkspaceSize(cutensorHandle* handle, cutensorOperationDescriptor* desc, cutensorPlanPreference* planPref, cutensorWorksizePreference_t workspacePref, ulong* workspaceSizeEstimate);

    [DllImport(__DllName, EntryPoint = "cutensorCreatePlan", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorCreatePlan(cutensorHandle* handle, cutensorPlan** plan, cutensorOperationDescriptor* desc, cutensorPlanPreference* pref, ulong workspaceSizeLimit);

    [DllImport(__DllName, EntryPoint = "cutensorDestroyPlan", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorDestroyPlan(cutensorPlan* plan);

    [DllImport(__DllName, EntryPoint = "cutensorContract", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorContract(cutensorHandle* handle, cutensorPlan* plan, void* alpha, void* A, void* B, void* beta, void* C, void* D, void* workspace, ulong workspaceSize, CUstream_st* stream);

    [DllImport(__DllName, EntryPoint = "cutensorCreateReduction", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorCreateReduction(cutensorHandle* handle, cutensorOperationDescriptor** desc, cutensorTensorDescriptor* descA, int* modeA, cutensorOperator_t opA, cutensorTensorDescriptor* descC, int* modeC, cutensorOperator_t opC, cutensorTensorDescriptor* descD, int* modeD, cutensorOperator_t opReduce, cutensorComputeDescriptor* descCompute);

    [DllImport(__DllName, EntryPoint = "cutensorReduce", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorReduce(cutensorHandle* handle, cutensorPlan* plan, void* alpha, void* A, void* beta, void* C, void* D, void* workspace, ulong workspaceSize, CUstream_st* stream);

    [DllImport(__DllName, EntryPoint = "cutensorGetErrorString", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern byte* cutensorGetErrorString(cutensorStatus_t error);

    [DllImport(__DllName, EntryPoint = "cutensorGetVersion", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern nuint cutensorGetVersion();

    [DllImport(__DllName, EntryPoint = "cutensorGetCudartVersion", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern nuint cutensorGetCudartVersion();

    [DllImport(__DllName, EntryPoint = "cutensorLoggerSetCallback", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorLoggerSetCallback(delegate* unmanaged[Cdecl]<int, byte*, byte*, void> callback);

    [DllImport(__DllName, EntryPoint = "cutensorLoggerSetFile", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorLoggerSetFile(_iobuf* file);

    [DllImport(__DllName, EntryPoint = "cutensorLoggerOpenFile", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorLoggerOpenFile(byte* logFile);

    [DllImport(__DllName, EntryPoint = "cutensorLoggerSetLevel", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorLoggerSetLevel(int level);

    [DllImport(__DllName, EntryPoint = "cutensorLoggerSetMask", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorLoggerSetMask(int mask);

    [DllImport(__DllName, EntryPoint = "cutensorLoggerForceDisable", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern cutensorStatus_t cutensorLoggerForceDisable();


}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct _iobuf
{
    public void* _Placeholder;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cutensorComputeDescriptor
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cutensorOperationDescriptor
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cutensorPlan
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cutensorPlanPreference
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cutensorHandle
{
    public fixed byte _unused[1];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe partial struct cutensorTensorDescriptor
{
    public fixed byte _unused[1];
}


public enum cutensorDataType_t : int
{
    CUTENSOR_R_16F = 2,
    CUTENSOR_C_16F = 6,
    CUTENSOR_R_16BF = 14,
    CUTENSOR_C_16BF = 15,
    CUTENSOR_R_32F = 0,
    CUTENSOR_C_32F = 4,
    CUTENSOR_R_64F = 1,
    CUTENSOR_C_64F = 5,
    CUTENSOR_R_4I = 16,
    CUTENSOR_C_4I = 17,
    CUTENSOR_R_4U = 18,
    CUTENSOR_C_4U = 19,
    CUTENSOR_R_8I = 3,
    CUTENSOR_C_8I = 7,
    CUTENSOR_R_8U = 8,
    CUTENSOR_C_8U = 9,
    CUTENSOR_R_16I = 20,
    CUTENSOR_C_16I = 21,
    CUTENSOR_R_16U = 22,
    CUTENSOR_C_16U = 23,
    CUTENSOR_R_32I = 10,
    CUTENSOR_C_32I = 11,
    CUTENSOR_R_32U = 12,
    CUTENSOR_C_32U = 13,
    CUTENSOR_R_64I = 24,
    CUTENSOR_C_64I = 25,
    CUTENSOR_R_64U = 26,
    CUTENSOR_C_64U = 27,
}

public enum cutensorOperator_t : int
{
    CUTENSOR_OP_IDENTITY = 1,
    CUTENSOR_OP_SQRT = 2,
    CUTENSOR_OP_RELU = 8,
    CUTENSOR_OP_CONJ = 9,
    CUTENSOR_OP_RCP = 10,
    CUTENSOR_OP_SIGMOID = 11,
    CUTENSOR_OP_TANH = 12,
    CUTENSOR_OP_EXP = 22,
    CUTENSOR_OP_LOG = 23,
    CUTENSOR_OP_ABS = 24,
    CUTENSOR_OP_NEG = 25,
    CUTENSOR_OP_SIN = 26,
    CUTENSOR_OP_COS = 27,
    CUTENSOR_OP_TAN = 28,
    CUTENSOR_OP_SINH = 29,
    CUTENSOR_OP_COSH = 30,
    CUTENSOR_OP_ASIN = 31,
    CUTENSOR_OP_ACOS = 32,
    CUTENSOR_OP_ATAN = 33,
    CUTENSOR_OP_ASINH = 34,
    CUTENSOR_OP_ACOSH = 35,
    CUTENSOR_OP_ATANH = 36,
    CUTENSOR_OP_CEIL = 37,
    CUTENSOR_OP_FLOOR = 38,
    CUTENSOR_OP_MISH = 39,
    CUTENSOR_OP_SWISH = 40,
    CUTENSOR_OP_SOFT_PLUS = 41,
    CUTENSOR_OP_SOFT_SIGN = 42,
    CUTENSOR_OP_ADD = 3,
    CUTENSOR_OP_MUL = 5,
    CUTENSOR_OP_MAX = 6,
    CUTENSOR_OP_MIN = 7,
    CUTENSOR_OP_UNKNOWN = 126,
}

public enum cutensorStatus_t : int
{
    CUTENSOR_STATUS_SUCCESS = 0,
    CUTENSOR_STATUS_NOT_INITIALIZED = 1,
    CUTENSOR_STATUS_ALLOC_FAILED = 3,
    CUTENSOR_STATUS_INVALID_VALUE = 7,
    CUTENSOR_STATUS_ARCH_MISMATCH = 8,
    CUTENSOR_STATUS_MAPPING_ERROR = 11,
    CUTENSOR_STATUS_EXECUTION_FAILED = 13,
    CUTENSOR_STATUS_INTERNAL_ERROR = 14,
    CUTENSOR_STATUS_NOT_SUPPORTED = 15,
    CUTENSOR_STATUS_LICENSE_ERROR = 16,
    CUTENSOR_STATUS_CUBLAS_ERROR = 17,
    CUTENSOR_STATUS_CUDA_ERROR = 18,
    CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
    CUTENSOR_STATUS_INSUFFICIENT_DRIVER = 20,
    CUTENSOR_STATUS_IO_ERROR = 21,
}

public enum cutensorAlgo_t : int
{
    CUTENSOR_ALGO_DEFAULT_PATIENT = -6,
    CUTENSOR_ALGO_GETT = -4,
    CUTENSOR_ALGO_TGETT = -3,
    CUTENSOR_ALGO_TTGT = -2,
    CUTENSOR_ALGO_DEFAULT = -1,
}

public enum cutensorWorksizePreference_t : int
{
    CUTENSOR_WORKSPACE_MIN = 1,
    CUTENSOR_WORKSPACE_DEFAULT = 2,
    CUTENSOR_WORKSPACE_MAX = 3,
}

public enum cutensorOperationDescriptorAttribute_t : int
{
    CUTENSOR_OPERATION_DESCRIPTOR_TAG = 0,
    CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE = 1,
    CUTENSOR_OPERATION_DESCRIPTOR_FLOPS = 2,
    CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES = 3,
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT = 4,
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT = 5,
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE = 6,
}

public enum cutensorPlanPreferenceAttribute_t : int
{
    CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE = 0,
    CUTENSOR_PLAN_PREFERENCE_CACHE_MODE = 1,
    CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT = 2,
    CUTENSOR_PLAN_PREFERENCE_ALGO = 3,
    CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK = 4,
    CUTENSOR_PLAN_PREFERENCE_JIT = 5,
}

public enum cutensorJitMode_t : int
{
    CUTENSOR_JIT_MODE_NONE = 0,
    CUTENSOR_JIT_MODE_DEFAULT = 1,
}

public enum cutensorPlanAttribute_t : int
{
    CUTENSOR_PLAN_REQUIRED_WORKSPACE = 0,
}