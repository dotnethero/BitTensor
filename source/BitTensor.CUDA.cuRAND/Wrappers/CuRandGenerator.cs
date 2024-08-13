using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using static cuRAND;

internal sealed unsafe class CuRandGenerator : IDisposable
{
    internal readonly curandGenerator_st* Generator;

    public CuRandGenerator()
    {
        curandGenerator_st* generator;

        var status = curandCreateGenerator(&generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
        Status.EnsureIsSuccess(status);

        Generator = generator;
    }

    public CuRandGenerator(ulong seed) : this()
    {
        var status = curandSetPseudoRandomGeneratorSeed(Generator, seed);
        Status.EnsureIsSuccess(status);
    }
    
    public void GenerateUniform(IDeviceArray<float> tensor)
    {
        var status = curandGenerateUniform(Generator, tensor.Pointer, (uint) tensor.Size);
        Status.EnsureIsSuccess(status);
        cudaRT.cudaThreadSynchronize();
    }

    public void GenerateNormal(IDeviceArray<float> tensor, float mean = 0f, float stddev = 1f)
    {
        var status = curandGenerateNormal(Generator, tensor.Pointer, (uint) tensor.Size, mean, stddev);
        Status.EnsureIsSuccess(status);
        cudaRT.cudaThreadSynchronize();
    }

    public void Dispose()
    {
        curandDestroyGenerator(Generator);
    }
}