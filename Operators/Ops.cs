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
    public static DenseTensor<float> ExpandDim(DenseTensor<float> tensor)
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
                    denseTensor[0, 0, i, x] = tensor[0, i, x];
                    denseTensor[0, 1, i, x] = tensor[1, i, x];
                    denseTensor[0, 2, i, x] = tensor[2, i, x];
                }
            });
        }

        return denseTensor;
    }

    /// <summary>
    /// only support 3 and 4 dimension
    /// </summary>
    /// <param name="denseTensor"></param>
    /// <returns></returns>
    public static DenseTensor<float> VerticalFlip(DenseTensor<float> denseTensor)
    {
        DenseTensor<float> tensor = new DenseTensor<float>(denseTensor.Dimensions);
        var dim = denseTensor.Dimensions.ToArray();
        if (dim.Length == 4)
        {
            for (int y = 0; y < dim[0]; y++)
            {
                Parallel.For(0, dim[2], i =>
                {
                    for (int x = 0; x < dim[3]; x++)
                    {
                        tensor[y, 0, i, x] = denseTensor[y, 0, dim[2] - 1 - i, x];
                        tensor[y, 1, i, x] = denseTensor[y, 1, dim[2] - 1 - i, x];
                        tensor[y, 2, i, x] = denseTensor[y, 2, dim[2] - 1 - i, x];
                    }
                });
            }
        }
        else
        {
            Parallel.For(0, dim[1], i =>
            {
                for (int x = 0; x < dim[2]; x++)
                {
                    tensor[0, i, x] = denseTensor[0, dim[2] - 1 - i, x];
                    tensor[1, i, x] = denseTensor[1, dim[2] - 1 - i, x];
                    tensor[2, i, x] = denseTensor[2, dim[2] - 1 - i, x];
                }
            });
        }

        return tensor;
    }

    /// <summary>
    /// only 3 and 4 dimension
    /// </summary>
    /// <param name="denseTensor"></param>
    /// <returns></returns>
    public static DenseTensor<float> HorizontalFlip(DenseTensor<float> denseTensor)
    {
        DenseTensor<float> tensor = new DenseTensor<float>(denseTensor.Dimensions);
        var dim = denseTensor.Dimensions.ToArray();
        if (dim.Length == 4)
        {
            for (int y = 0; y < dim[0]; y++)
            {
                Parallel.For(0, dim[1], i =>
                {
                    for (int x = 0; x < dim[3]; x++)
                    {
                        tensor[y, 0, i, x] = denseTensor[y, 0, i, dim[3] - 1 - x];
                        tensor[y, 1, i, x] = denseTensor[y, 1, i, dim[3] - 1 - x];
                        tensor[y, 2, i, x] = denseTensor[y, 2, i, dim[3] - 1 - x];
                    }
                });
            }
        }
        else
        {
            Parallel.For(0, dim[2], i =>
            {
                for (int x = 0; x < dim[2]; x++)
                {
                    tensor[0, i, x] = denseTensor[0, i, dim[2] - 1 - x];
                    tensor[1, i, x] = denseTensor[1, i, dim[2] - 1 - x];
                    tensor[2, i, x] = denseTensor[2, i, dim[2] - 1 - x];
                }
            });
        }

        return tensor;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensors"></param>
    /// <returns></returns>
    public static DenseTensor<float> Concat(List<DenseTensor<float>> tensors)
    {
        int[] fim_first = tensors.First().Dimensions.ToArray();
        int[] dim = new[] { tensors.Count, fim_first[0], fim_first[1], fim_first[2] };
        DenseTensor<float> value = new DenseTensor<float>(tensors.Count);

        Parallel.For(0, fim_first[1], i =>
        {
            for (int x = 0; x < fim_first[2]; x++)
            {
                for (int y = 0; y < tensors.Count; y++)
                {
                    value[y, 0, x, i] = tensors[y][0, x, i];
                    value[y, 1, x, i] = tensors[y][1, x, i];
                    value[y, 2, x, i] = tensors[y][2, x, i];
                }
            }
        });

        return value;
    }
}