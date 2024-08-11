﻿using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

using Ops = cutensorOperator_t;

public sealed class CuTensorBroadcastPlan : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;
    internal readonly CuTensorBinaryOperation Operation;
    internal readonly CuTensorPlan OperationPlan;

    internal CuTensorBroadcastPlan(
        CuTensorContext context,
        AbstractTensor a,
        AbstractTensor b)
    {
        LeftDescriptor = context.CreateDescriptor(a);
        RightDescriptor = context.CreateDescriptor(b);

        Operation = new CuTensorBinaryOperation(
            context,
            LeftDescriptor,
            RightDescriptor,
            RightDescriptor,
            Ops.CUTENSOR_OP_IDENTITY,
            Ops.CUTENSOR_OP_IDENTITY,
            Ops.CUTENSOR_OP_ADD);

        OperationPlan = Operation.CreatePlan();
    }
    
    public void Execute(CuTensor left, CuTensor right, float alpha = 1f, float gamma = 1f) =>
        Operation.Execute(
            OperationPlan,
            left,
            right,
            right,
            alpha,
            gamma);

    public void Dispose()
    {
        OperationPlan.Dispose();
        Operation.Dispose();
        RightDescriptor.Dispose();
        LeftDescriptor.Dispose();
    }
}