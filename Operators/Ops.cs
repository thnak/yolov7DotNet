using Microsoft.ML.OnnxRuntime.Tensors;

namespace yolov7DotNet.Operators;

public class Operators
{
    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="maxPixelValue"></param>
    /// <returns></returns>
    public static DenseTensor<float> Div(DenseTensor<float> tensor, float maxPixelValue = 255f)
    {
        int[] dim = tensor.Dimensions.ToArray();
        Parallel.For(0, dim[1], x =>
        {
            for (int y = 0; y < dim[2]; y++)
            {
                tensor[0, x, y] /= maxPixelValue;
                tensor[1, x, y] /= maxPixelValue;
                tensor[2, x, y] /= maxPixelValue;
            }
        });
        return tensor;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    public static DenseTensor<float> Mul(DenseTensor<float> tensor, float value = 255f)
    {
        int[] dim = tensor.Dimensions.ToArray();
        Parallel.For(0, dim[1], x =>
        {
            for (int y = 0; y < dim[2]; y++)
            {
                tensor[0, x, y] *= value;
                tensor[1, x, y] *= value;
                tensor[2, x, y] *= value;
            }
        });
        return tensor;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    public static DenseTensor<float> Add(DenseTensor<float> tensor, float value = 255f)
    {
        int[] dim = tensor.Dimensions.ToArray();
        Parallel.For(0, dim[1], x =>
        {
            for (int y = 0; y < dim[2]; y++)
            {
                tensor[0, x, y] += value;
                tensor[1, x, y] += value;
                tensor[2, x, y] += value;
            }
        });
        return tensor;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    public static DenseTensor<float> Add(DenseTensor<float> tensor1, DenseTensor<float> tensor2)
    {
        int[] dim = tensor1.Dimensions.ToArray();
        Parallel.For(0, dim[1], x =>
        {
            for (int y = 0; y < dim[2]; y++)
            {
                tensor1[0, x, y] += tensor2[0, x, y];
                tensor1[1, x, y] += tensor2[1, x, y];
                tensor1[2, x, y] += tensor2[2, x, y];
            }
        });
        return tensor1;
    }
}

