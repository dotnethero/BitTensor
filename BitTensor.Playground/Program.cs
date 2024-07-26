using System.Diagnostics;
using BitTensor.Abstractions;
using BitTensor.Core;
using BitTensor.CUDA;
using BitTensor.CUDA.Interop;
using static BitTensor.CUDA.Interop.cudaRT;
using static BitTensor.CUDA.Interop.cudaMemcpyKind;
using static BitTensor.CUDA.Interop.cuBLAS;
using static BitTensor.CUDA.Interop.cublasStatus_t;
using static BitTensor.CUDA.Interop.cublasOperation_t;

namespace BitTensor.Playground;

internal static unsafe class CuTensorOps
{
    public static void Multiply(Tensor a, Tensor b, Tensor result)
    {
        cublasContext* handle;

        var status = cublasCreate_v2(&handle);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw new InvalidOperationException(status.ToString());

        var da = ((DebugDeviceAllocation)a.Allocation).Pointer;
        var db = ((DebugDeviceAllocation)b.Allocation).Pointer;
        var dc = ((DebugDeviceAllocation)result.Allocation).Pointer;

        float alpha = 1.0f;
        float beta = 0.0f;
        const int incx = 1;
        const int incy = 1;
        cublasSsbmv_v2(handle, cublasFillMode_t.CUBLAS_FILL_MODE_UPPER, a.Size, 0, &alpha, da, 1, db, incx, &beta, dc, incy);

        cublasDestroy_v2(handle);
    }
}

internal unsafe class Program
{
    static void Main(string[] args)
    {
        using var a = CuTensor.Create([1, 2, 3]);
        using var b = CuTensor.Create([3, 4, 5]);
        using var c = a * b;

        var data = new float[c.Size];

        c.EnsureHasUpdatedValues();
        c.CopyToHost(data);

        var output = Tensor.FromArray(c.Shape, data);

        Console.WriteLine(output.ToDataString());
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

        cudaMalloc((void**)&da, mu * nu * sizeof(float));
        cudaMalloc((void**)&db, nu * ku * sizeof(float));
        cudaMalloc((void**)&dc, mu * ku * sizeof(float)); // column-major result
        cudaMalloc((void**)&dr, mu * ku * sizeof(float));

        fixed (float*
               va = a.Values,
               vb = b.Values)
        {
            cudaMemcpy(da, va, mu * nu * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(db, vb, nu * ku * sizeof(float), cudaMemcpyHostToDevice);
        }

        var one = 1.0f;
        var zero = 0.0f;

        var sw = Stopwatch.StartNew();

        cublasSgemm_v2(
            context, 
            CUBLAS_OP_T,
            CUBLAS_OP_T,
            mi,
            ki,
            ni,
            &one, 
            da, ni, // m * n
            db, ki, // n * k
            &zero, 
            dc, mi); // m * k

        cublasSgeam(
            context,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
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

        cudaDeviceSynchronize();

        fixed (float* vc = r.Data)
        {
            cudaMemcpy(vc, dr, mu * ku * sizeof(float), cudaMemcpyDeviceToHost);
        }

        var s = sw.Elapsed;
        var gflops = (2 / s.TotalSeconds * mi * ni * ki) / 1e9;

        Console.WriteLine($"{gflops:0.00} GFLOP/s");
        Console.WriteLine($"{s} to result");

        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);

        cublasDestroy_v2(context);
    }

    private static cublasContext* CreateContext()
    {
        cublasContext* handle;

        var status = cublasCreate_v2(&handle);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw new InvalidOperationException(status.ToString());

        return handle;
    }
}