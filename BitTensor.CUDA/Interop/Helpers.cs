#nullable disable
#pragma warning disable CS8500, CS8981

using System.Runtime.InteropServices;

namespace BitTensor.CUDA.Interop;

internal static class Helpers
{
    public static IntPtr ReadConstant(IntPtr lib, string name)
    {
        var export = NativeLibrary.GetExport(lib, name);
        var pointer = Marshal.ReadIntPtr(export);
        return pointer;
    }
}