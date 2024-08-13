using System.Runtime.InteropServices;

namespace BitTensor.CUDA.Interop;

public static unsafe class Kernels
{
    // ReSharper disable once InconsistentNaming
    private const string __DllName = "BitTensor.CUDA.Kernels.dll";
    
    [DllImport(__DllName, EntryPoint = "f32_leaky_relu", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern void LeakyReLU(int size, float* a, float* output, float alpha, CUstream_st* stream = default);
}
