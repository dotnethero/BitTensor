using System.Diagnostics;
using BitTensor.Abstractions;
using BitTensor.Core;
using BitTensor.Native;

namespace BitTensor.Playground;

internal static class CuTensor
{
    public static Tensor FromArray(int[] shape, float[] values)
    {
        var allocation = DebugDeviceAllocator.Instance.Allocate(values.Length);
        allocation.CopyToDevice(values);
        return new Tensor(shape, allocation);
    }

    public static Tensor Create(float[] values) =>
        FromArray(shape: [values.Length], values);

    public static Tensor Mul(Tensor a, Tensor b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Shape, b.Shape);
        
        return new(
            shape,
            children: [a, b],
            forward: static self => CuTensorOps.Multiply(self.A, self.B, self),
            backward: static (grad, self) => [self.B * grad, self.A * grad],
            allocator: DebugDeviceAllocator.Instance);
    }
}

internal static unsafe class CuTensorOps
{
    public static void Multiply(Tensor a, Tensor b, Tensor result)
    {
        cublasContext* handle;

        var status = cuBLAS.cublasCreate_v2(&handle);
        if (status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            throw new InvalidOperationException(status.ToString());

        var da = ((DebugDeviceAllocation)a.Allocation).Pointer;
        var db = ((DebugDeviceAllocation)b.Allocation).Pointer;
        var dc = ((DebugDeviceAllocation)result.Allocation).Pointer;

        float alpha = 1.0f;
        float beta = 0.0f;
        const int incx = 1;
        const int incy = 1;
        cuBLAS.cublasSsbmv_v2(handle, cublasFillMode_t.CUBLAS_FILL_MODE_UPPER, a.Size, 0, &alpha, da, 1, db, incx, &beta, dc, incy);

        cuBLAS.cublasDestroy_v2(handle);
    }
}

internal unsafe class Program
{
    static void Main(string[] args)
    {
        var a = CuTensor.Create([1, 2, 3]);
        var b = CuTensor.Create([3, 4, 5]);
        var c = CuTensor.Mul(a, b);

        Console.WriteLine(c.ToDataString());
    }

    private static void GemmExample()
    {
        const int m = 1024 * 10;
        const int n = 1024 * 10;
        const int k = 1024 * 10;

        for (var i = 0; i < 4; i++)
        {
            var a = Tensor.Random.Uniform([m, n]);
            var b = Tensor.Random.Uniform([n, k]);
            var c = Tensor.Zeros([m, k]);
            var d = Tensor.Matmul(a, b);
            var sw = Stopwatch.StartNew();
            Gemm(a, b, c);
            Console.WriteLine($"{sw.Elapsed} total");
        }
    }
        
    private static void Gemm(Tensor a, Tensor b, Tensor r)
    {
        var context = CreateContext();

        var mi = a.PrevDimension;
        var ni = a.LastDimension;
        var ki = b.LastDimension;

        var mu = (nuint) mi;
        var nu = (nuint) ni;
        var ku = (nuint) ki;

        float* da;
        float* db;
        float* dc;
        float* dr;

        CUDA.cudaMalloc((void**)&da, mu * nu * sizeof(float));
        CUDA.cudaMalloc((void**)&db, nu * ku * sizeof(float));
        CUDA.cudaMalloc((void**)&dc, mu * ku * sizeof(float)); // column-major result
        CUDA.cudaMalloc((void**)&dr, mu * ku * sizeof(float));

        fixed (float*
               va = a.Values,
               vb = b.Values)
        {
            CUDA.cudaMemcpy(da, va, mu * nu * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);
            CUDA.cudaMemcpy(db, vb, nu * ku * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);
        }

        var one = 1.0f;
        var zero = 0.0f;

        var sw = Stopwatch.StartNew();

        cuBLAS.cublasSgemm_v2(
            context, 
            cublasOperation_t.CUBLAS_OP_T,
            cublasOperation_t.CUBLAS_OP_T,
            mi,
            ki,
            ni,
            &one, 
            da, ni, // m * n
            db, ki, // n * k
            &zero, 
            dc, mi); // m * k

        cuBLAS.cublasSgeam(
            context,
            cublasOperation_t.CUBLAS_OP_T,
            cublasOperation_t.CUBLAS_OP_N,
            mi,
            ki,
            &one,
            dc,
            mi,
            &zero,
            (float*)0, 
            ki,
            dr,
            ki);

        CUDA.cudaDeviceSynchronize();

        fixed (float* vc = r.Data)
        {
            CUDA.cudaMemcpy(vc, dr, mu * ku * sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost);
        }

        var s = sw.Elapsed;
        var gflops = (2 / s.TotalSeconds * mi * ni * ki) / 1e9;

        Console.WriteLine($"{gflops:0.00} GFLOP/s");
        Console.WriteLine($"{s} to result");

        CUDA.cudaFree(da);
        CUDA.cudaFree(db);
        CUDA.cudaFree(dc);

        cuBLAS.cublasDestroy_v2(context);
    }

    private static cublasContext* CreateContext()
    {
        cublasContext* handle;

        var status = cuBLAS.cublasCreate_v2(&handle);
        if (status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            throw new InvalidOperationException(status.ToString());

        return handle;
    }
}