using System.Numerics;
using System.Runtime.Intrinsics;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace yolov7DotNet.Operators;

public abstract class SIMDTensor
{
    public static float[] Div(DenseTensor<float> tensor1, float maxPixelValue)
    {
        var array = tensor1.ToArray();
        int remaining = array.Length % Vector<double>.Count;
        
        var array2 = new float[array.Length];
        Array.Fill(array, maxPixelValue);
        var result = new float[array.Length];

        for (var x = 0; x < array.Length - remaining; x += Vector<float>.Count)
        {
            var v1 = new Vector<float>(array, x);
            var v2 = new Vector<float>(array2, x);
            (v1 / v2).CopyTo(result, x);
        }

        for (var x = array.Length - remaining; x < array.Length; x++)
        {
            result[x] = array[x] / array2[x];
        }
        return result;
    }
}