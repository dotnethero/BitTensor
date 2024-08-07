namespace BitTensor.Abstractions;

public class NotCompatibleShapesException(int[] shape1, int[] shape2) : 
    Exception($"Shapes are incompatible: {shape1.Serialize()} and {shape2.Serialize()}");
