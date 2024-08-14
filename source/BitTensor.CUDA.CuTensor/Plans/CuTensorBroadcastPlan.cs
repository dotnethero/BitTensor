﻿using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

using Ops = cutensorOperator_t;

public sealed class CuTensorBroadcastPlan<T> : ICuTensorPlan where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CuTensorDescriptor<T> LeftDescriptor;
    internal readonly CuTensorDescriptor<T> RightDescriptor;
    internal readonly CuTensorBinaryOperation<T> Operation;
    internal readonly CuTensorPlan OperationPlan;
    internal bool IsDisposed;

    internal CuTensorBroadcastPlan(
        CuTensorContext context,
        Shape a,
        Shape b)
    {
        LeftDescriptor = new(context, a);
        RightDescriptor = new(context, b);

        Operation = new(
            context,
            LeftDescriptor,
            RightDescriptor,
            RightDescriptor,
            Ops.CUTENSOR_OP_IDENTITY,
            Ops.CUTENSOR_OP_IDENTITY,
            Ops.CUTENSOR_OP_ADD);

        OperationPlan = Operation.CreatePlan();
    }
    
    public void Execute(IDeviceArray<T> left, IDeviceArray<T> right, float alpha = 1f, float gamma = 0f) =>
        Operation.Execute(
            OperationPlan,
            left,
            right,
            right,
            alpha,
            gamma);

    public void Dispose()
    {
        if (IsDisposed) return;

        OperationPlan.Dispose();
        Operation.Dispose();
        RightDescriptor.Dispose();
        LeftDescriptor.Dispose();
        IsDisposed = true;
    }
}