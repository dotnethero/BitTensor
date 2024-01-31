using System.Diagnostics;
using BitTensor.Core;
using BitTensor.Native;

namespace BitTensor.Playground
{
    internal unsafe class Program
    {
        static void Main(string[] args)
        {
            GemmExample();
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
}
