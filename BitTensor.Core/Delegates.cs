using BitTensor.Abstractions;

namespace BitTensor.Core;

/// <summary>
/// Function that retrieves gradients of expression with respect to specific variables
/// </summary>
/// <param name="variables">Variables for partial gradient</param>
public delegate T[] GetGradientsFunction<T>(IEnumerable<T> variables) where T : AbstractTensor, ITensor<T>;
