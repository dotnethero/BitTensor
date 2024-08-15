using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using OpCode = cutensorOperator_t;

public class Operand
{
    internal readonly Shape Shape;
    internal readonly OpCode Transformation;

    public static implicit operator Operand(Shape shape) => new(shape);
    
    public static Operand Exp(Shape shape) => new(shape, OpCode.CUTENSOR_OP_EXP);
    public static Operand Rcp(Shape shape) => new(shape, OpCode.CUTENSOR_OP_RCP);
    public static Operand Relu(Shape shape) => new(shape, OpCode.CUTENSOR_OP_RELU);

    private Operand(Shape shape)
    {
        Shape = shape;
        Transformation = OpCode.CUTENSOR_OP_IDENTITY;
    }
    
    private Operand(Shape shape, OpCode transformation)
    {
        Shape = shape;
        Transformation = transformation;
    }
}
