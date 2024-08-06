﻿using System.Diagnostics;
using System.Runtime.CompilerServices;
using BitTensor.CUDA.ComputeOnly.Plans;
using BitTensor.CUDA.ComputeOnly.Wrappers;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly;

internal class Program
{
    public static void Main()
    {
        const int B = 1024;
        const int N = 128;
        const int K = 512;

        using var a = CuTensor.Random.Uniform([B, N, K]);
        using var b = CuTensor.Random.Uniform([B, N, K]);

        using var z = CuTensor.Allocate([B, N, K]);

        BenchAdd(() => CuBLAS.Add(a, b, z));
        BenchAdd(() => CuBLAS.Add(a, b, z));
        BenchAdd(() => CuBLAS.Add(a, b, z));
        
        BenchAdd(() => CuTensor.Add(a, b, z));
        BenchAdd(() => CuTensor.Add(a, b, z));
        BenchAdd(() => CuTensor.Add(a, b, z));

        using var context = new CuTensorContext();
        using var plan = new CuTensorElementwiseAdd(context, a, b, z);

        BenchAdd(() => plan.Execute(a, b, z));
        BenchAdd(() => plan.Execute(a, b, z));
        BenchAdd(() => plan.Execute(a, b, z));

        return;

        void BenchAdd(Action action, [CallerArgumentExpression("action")] string actionName = "")
        {
            var sw = Stopwatch.StartNew();
            action();
            cudaRT.cudaDeviceSynchronize();

            var flops = (B * N * K / sw.Elapsed.TotalSeconds) / 1e9;
            Console.WriteLine($"{actionName}: {sw.Elapsed}, {flops} GFLOPs");
        }
    }
}