using Microsoft.ML.OnnxRuntime.Tensors;

namespace yolov7DotNet.Operators;

public class Operators
{
    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor1"></param>
    /// <param name="maxPixelValue"></param>
    /// <returns></returns>
    public static DenseTensor<float> Div(DenseTensor<float> tensor1, float maxPixelValue)
    {
        int[] dim = tensor1.Dimensions.ToArray();
        Parallel.For(0, dim[1], x =>
        {
            for (int y = 0; y < dim[2]; y++)
            {
                tensor1[0, x, y] /= maxPixelValue;
                tensor1[1, x, y] /= maxPixelValue;
                tensor1[2, x, y] /= maxPixelValue;
            }
        });
        return tensor1;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor1"></param>
    /// <param name="tensor2"></param>
    /// <returns></returns>
    public static DenseTensor<float> Div(DenseTensor<float> tensor1, DenseTensor<float> tensor2)
    {
        int[] dim = tensor1.Dimensions.ToArray();
        Parallel.For(0, dim[1], x =>
        {
            for (int y = 0; y < dim[2]; y++)
            {
                tensor1[0, x, y] /= tensor2[0, x, y];
                tensor1[1, x, y] /= tensor2[1, x, y];
                tensor1[2, x, y] /= tensor2[2, x, y];
            }
        });
        return tensor1;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor1"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    public static DenseTensor<float> Mul(DenseTensor<float> tensor1, float value)
    {
        int[] dim = tensor1.Dimensions.ToArray();
        Parallel.For(0, dim[1], x =>
        {
            for (int y = 0; y < dim[2]; y++)
            {
                tensor1[0, x, y] *= value;
                tensor1[1, x, y] *= value;
                tensor1[2, x, y] *= value;
            }
        });
        return tensor1;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor1"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    public static DenseTensor<float> Mul(Tensor<float> tensor1, float value)
    {
        int[] dim = tensor1.Dimensions.ToArray();
        Parallel.For(0, dim[1], x =>
        {
            for (int y = 0; y < dim[2]; y++)
            {
                tensor1[0, x, y] *= value;
                tensor1[1, x, y] *= value;
                tensor1[2, x, y] *= value;
            }
        });
        return tensor1.ToDenseTensor();
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor1"></param>
    /// <param name="tensor2"></param>
    /// <returns></returns>
    public static DenseTensor<float> Mul(DenseTensor<float> tensor1, DenseTensor<float> tensor2)
    {
        int[] dim = tensor1.Dimensions.ToArray();
        Parallel.For(0, dim[1], x =>
        {
            for (int y = 0; y < dim[2]; y++)
            {
                tensor1[0, x, y] *= tensor2[0, x, y];
                tensor1[1, x, y] *= tensor2[0, x, y];
                tensor1[2, x, y] *= tensor2[0, x, y];
            }
        });
        return tensor1;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    public static DenseTensor<float> Add(DenseTensor<float> tensor, float value)
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
    /// <param name="tensor1"></param>
    /// <param name="tensor2"></param>
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
    /// <summary>
    /// implement of numpy expandim
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    internal static DenseTensor<float> ExpandDim(DenseTensor<float> tensor)
    {
        int height = tensor.Dimensions[1];
        int width = tensor.Dimensions[2];
        int[] shape = new[] { 1, 3, height, width };

        DenseTensor<float> denseTensor = new DenseTensor<float>(shape);

        if (height == width)
        {
            Parallel.For(0, height, i =>
            {
                denseTensor[0, 0, i, i] = tensor[0, i, i];
                denseTensor[0, 1, i, i] = tensor[1, i, i];
                denseTensor[0, 2, i, i] = tensor[2, i, i];
            });
        }
        else
        {
            Parallel.For(0, height, i =>
            {
                for (int x = 0; x < width; x++)
                {
                    denseTensor[0, 1, i, x] = tensor[1, i, x];
                    denseTensor[0, 2, i, x] = tensor[2, i, x];
                    denseTensor[0, 3, i, x] = tensor[3, i, x];
                }
            });
        }

        return denseTensor;
    }
}