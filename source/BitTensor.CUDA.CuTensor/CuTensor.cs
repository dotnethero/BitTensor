using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BitTensor.Abstractions;

namespace BitTensor.CUDA;

public class CuTensor : AbstractTensor
{
    public CuTensor(Shape shape) : base(shape)
    {
    }
}