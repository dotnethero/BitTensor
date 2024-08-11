﻿using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal sealed class CublasException(cublasStatus_t status) : Exception($"Operation is not completed: {status}");