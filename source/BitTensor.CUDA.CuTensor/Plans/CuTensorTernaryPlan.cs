﻿using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

public sealed class CuTensorTernaryPlan<T> : ICuTensorPlan where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CuTensorDescriptor<T> LeftDescriptor;
    internal readonly CuTensorDescriptor<T> RightDescriptor;
    internal readonly CuTensorDescriptor<T> ResultDescriptor;
    internal readonly CuTensorTernaryOperation<T> Operation;
    internal readonly CuTensorPlan OperationPlan;
    internal bool IsDisposed;

    internal CuTensorTernaryPlan(
        CuTensorContext context,
        Operand a,
        Operand b,
        Operand c,
        cutensorOperator_t opAB,
        cutensorOperator_t opABC)
    {
        LeftDescriptor = new(context, a);
        RightDescriptor = new(context, b);
        ResultDescriptor = new(context, c);

        Operation = new(
            context,
            LeftDescriptor,
            RightDescriptor,
            ResultDescriptor,
            ResultDescriptor,
            opAB,
            opABC);

        OperationPlan = Operation.CreatePlan();
    }
    
    public void Execute(
        IDeviceArray<T> left,
        IDeviceArray<T> right,
        IDeviceArray<T> result,
        float alpha = 1f,
        float beta = 1f,
        float gamma = 0f) =>
        Operation.Execute(
            OperationPlan,
            left,
            right,
            result,
            result,
            alpha,
            beta,
            gamma);

    public void Dispose()
    {
        if (IsDisposed) return;

        OperationPlan.Dispose();
        Operation.Dispose();
        ResultDescriptor.Dispose();
        RightDescriptor.Dispose();
        LeftDescriptor.Dispose();
        IsDisposed = true;
    }
}