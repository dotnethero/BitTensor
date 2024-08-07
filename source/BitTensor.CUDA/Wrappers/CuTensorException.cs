﻿using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal class CuTensorException(cutensorStatus_t status) : Exception($"Operation is not completed: {status}");