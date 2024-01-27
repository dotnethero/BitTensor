using BitTensor.Core;
using BitTensor.Native;

namespace BitTensor.Playground
{
    internal unsafe class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            GemmExample();
        }

        private static void GemmExample()
        {
            const int m = 2;
            const int n = 3;
            const int k = 4;

            var a = Tensor.Random.Uniform([m, n]);
            var b = Tensor.Random.Uniform([n, k]);
            var c = Tensor.Zeros([m, k]);
            var d = Tensor.Matmul(a, b).T;

            Gemm(a, b, c);

            var cr = c.Values;
            var dr = d.Values;
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

            CUDA.cudaMalloc((void**)&da, mu * nu * sizeof(float));
            CUDA.cudaMalloc((void**)&db, nu * ku * sizeof(float));
            CUDA.cudaMalloc((void**)&dc, mu * ku * sizeof(float));

            fixed (float*
                   va = a.Values,
                   vb = b.Values)
            {
                CUDA.cudaMemcpy(da, va, mu * nu * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);
                CUDA.cudaMemcpy(db, vb, nu * ku * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);
            }

            var alpha = 1.0f;
            var beta = 0.0f;

            cuBLAS.cublasSgemm_v2(
                context, 
                cublasOperation_t.CUBLAS_OP_T,
                cublasOperation_t.CUBLAS_OP_T,
                mi,
                ki,
                ni,
                &alpha, 
                da, ni, // m * n
                db, ki, // n * k
                &beta, 
                dc, mi); // m * k

            fixed (float* vc = r.Data)
            {
                CUDA.cudaMemcpy(vc, dc, mu * ku * sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost);
            }

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
