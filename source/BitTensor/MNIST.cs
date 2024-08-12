using BitTensor.Abstractions;

namespace BitTensor;

public static unsafe class MNIST
{
    private delegate void Epilogue<T>(Span<T> array, ReadOnlySpan<byte> buffer);

    public static Dataset<float> ReadImages(string path)
    {
        using var stream = File.OpenRead(path);
        var samples = ReadFile<float>(stream, Normalize);
        return samples;
    }

    public static Dataset<float> ReadLabels(string path, int classes = 10)
    {
        var samples = ReadLabelsRaw(path);
        var shape = samples.Shape.Append(classes);
        var onehot = new float[shape.ArraySize];

        fixed (byte* src = samples.Data)
        fixed (float* dst = onehot)
        {
            for (var i = 0; i < samples.Data.Length; ++i)
            {
                dst[i * classes + src[i]] = 1;
            }
        }
        return new(shape, onehot);
    }
    
    private static Dataset<byte> ReadLabelsRaw(string path)
    {
        using var stream = File.OpenRead(path);
        var samples = ReadFile<byte>(stream, CopyToView);
        return samples;
    }

    private static void Normalize(Span<float> view, ReadOnlySpan<byte> buffer)
    {
        fixed (byte* src = buffer)
        fixed (float* dst = view)
        {
            for (var i = 0; i < view.Length; ++i)
            {
                dst[i] = src[i] / 255f;
            }
        }
    }

    private static void CopyToView(Span<byte> view, ReadOnlySpan<byte> buffer)
    {
        buffer.CopyTo(view);
    }

    private static Dataset<T> ReadFile<T>(Stream stream, Epilogue<T> epilogue)
    {
        var magic = new byte[4];
        stream.ReadExactly(magic);
        
        var elementSize = GetElementSize(magic[2]);
        var dimensions = ReadDimensions(stream, magic[3]);
        var samples = ReadSamples(stream, dimensions, elementSize, epilogue);

        return samples;
    }

    private static Dataset<T> ReadSamples<T>(Stream stream, int[] dimensions, int elementSize, Epilogue<T> epilogue)
    {
        var shape = Shape.Create(dimensions);
        var samples = shape[0];
        var sampleStride = shape.Strides[0];

        var array = new T[shape.ArraySize];
        var buffer = new byte[sampleStride * elementSize];

        for (var i = 0; i < samples; i++)
        {
            stream.ReadExactly(buffer);
            var view = array.AsSpan(i * sampleStride, sampleStride);
            epilogue(view, buffer);
        }
        return new(shape, array);
    }
    
    // 0x08: unsigned byte
    // 0x09: signed byte
    // 0x0B: short (2 bytes)
    // 0x0C: int (4 bytes)
    // 0x0D: float (4 bytes)
    // 0x0E: double (8 bytes)

    private static int GetElementSize(byte type) =>
        type switch
        {
            0x08 or 0x09 => 1,
            0x0B => 2,
            0x0C or 0x0D => 4,
            0x0E => 8,
            _ => throw new InvalidOperationException($"Unsupported type: {type:X}")
        };

    private static int[] ReadDimensions(Stream stream, byte dims)
    {
        var sizesBytes = new byte[dims << 2];
        stream.ReadExactly(sizesBytes);

        var sizes = new int[dims];
        for (var i = 0; i < dims; ++i)
        {
            var b = sizesBytes.AsSpan(i << 2, 4);
            sizes[i] = b[0] << 24 | b[1] << 16 | b[2] << 8 | b[3];
        }
        return sizes;
    }
}