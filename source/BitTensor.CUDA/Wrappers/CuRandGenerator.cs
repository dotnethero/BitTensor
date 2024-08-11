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
    
    public CuTensor GenerateUniform(CuTensor tensor)
    {
        var status = curandGenerateUniform(Generator, tensor.Pointer, (uint) tensor.Size);
        Status.EnsureIsSuccess(status);
        cudaRT.cudaThreadSynchronize();
        return tensor;
    }

    public CuTensor GenerateNormal(CuTensor tensor, float mean = 0f, float stddev = 1f)
    {
        var status = curandGenerateNormal(Generator, tensor.Pointer, (uint) tensor.Size, mean, stddev);
        Status.EnsureIsSuccess(status);
        cudaRT.cudaThreadSynchronize();
        return tensor;
    }

    public void Dispose()
    {
        curandDestroyGenerator(Generator);
    }
}