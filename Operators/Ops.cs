using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace yolov7DotNet.Operators;

public abstract class Ops
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
        DenseTensor<float> tensor3 = new DenseTensor<float>(tensor1.Dimensions);

        Parallel.For(0, tensor1.Dimensions[0], i =>
        {
            for (int j = 0; j < tensor1.Dimensions[1]; j++)
            {
                for (int k = 0; k < tensor1.Dimensions[1]; k++)
                {
                    tensor3[i, j] += tensor1[i, k] * tensor2[k, j];
                }
            }
        });

        return tensor3;
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
        int ndim = tensor1.Rank;
        switch (ndim)
        {
            case 2:
            {
                for (int x = 0; x < dim[0]; x++)
                {
                    for (int y = 0; y < dim[1]; y++)
                    {
                        tensor1[x, y] += tensor2[x, y];
                    }
                }

                break;
            }
            case 3:
            {
                Parallel.For(0, dim[1], x =>
                {
                    for (int y = 0; y < dim[2]; y++)
                    {
                        tensor1[0, x, y] += tensor2[0, x, y];
                        tensor1[1, x, y] += tensor2[1, x, y];
                        tensor1[2, x, y] += tensor2[2, x, y];
                    }
                });
                break;
            }
        }

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

        DenseTensor<float> denseTensor = new DenseTensor<float>(dimensions: shape);
        Parallel.For(0, height, i =>
        {
            Parallel.For(0, width, x =>
            {
                denseTensor[0, 0, i, x] = tensor[0, i, x];
                denseTensor[0, 1, i, x] = tensor[1, i, x];
                denseTensor[0, 2, i, x] = tensor[2, i, x];
            });
        });

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
    /// concatenate 3D array into 4D array
    /// </summary>
    /// <param name="tensors"></param>
    /// <returns></returns>
    public static DenseTensor<float> Concat(List<DenseTensor<float>> tensors)
    {
        int[] fim_first = tensors.First().Dimensions.ToArray();
        int[] dim = new[] { tensors.Count, fim_first[0], fim_first[1], fim_first[2] };
        DenseTensor<float> value = new DenseTensor<float>(dim);

        Parallel.For(0, fim_first[1], x =>
        {
            for (int i = 0; i < fim_first[2]; i++)
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

    /// <summary>
    /// convert float32 tensor to float16 tensor
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static DenseTensor<Float16> ToHalfTensor(DenseTensor<float> tensor)
    {
        DenseTensor<Float16> halfTensor = new DenseTensor<Float16>(tensor.Dimensions);
        var rank = tensor.Rank;
        if (rank == 3)
        {
            Parallel.For(0, tensor.Dimensions[1], i =>
            {
                Parallel.For(0, tensor.Dimensions[2], x =>
                {
                    halfTensor[0, i, x] = (Float16)tensor[0, i, x];
                    halfTensor[1, i, x] = (Float16)tensor[1, i, x];
                    halfTensor[2, i, x] = (Float16)tensor[2, i, x];
                });
            });
        }
        else
        {
            Parallel.For(0, tensor.Dimensions[2], i =>
            {
                Parallel.For(0, tensor.Dimensions[3], x =>
                {
                    for (int y = 0; y < tensor.Dimensions[0]; y++)
                    {
                        halfTensor[y, 0, i, x] = (Float16)tensor[y, 0, i, x];
                        halfTensor[y, 1, i, x] = (Float16)tensor[y, 1, i, x];
                        halfTensor[y, 2, i, x] = (Float16)tensor[y, 2, i, x];
                    }
                });
            });
        }

        return halfTensor;
    }

    public static DenseTensor<float> BitMap2Tensor(Bitmap bitmap)
    {
        int[] shape = new[] { 3, bitmap.Height, bitmap.Width };
        DenseTensor<float> tensor = new DenseTensor<float>(shape);

        Parallel.For(0, bitmap.Height, i =>
        {
            for (int x = 0; x < bitmap.Width; x++)
            {
                tensor[0, i, x] = bitmap.GetPixel(x, i).R;
                tensor[1, i, x] = bitmap.GetPixel(x, i).G;
                tensor[2, i, x] = bitmap.GetPixel(x, i).B;
            }
        });
        Tensor<float> a = tensor;
        var c = a[0, 1, 2];
        return tensor;
    }

    /// <summary>
    /// numpy implement of eye
    /// </summary>
    /// <param name="shape"></param>
    /// <returns>ma trận đường chéo</returns>
    public static DenseTensor<float> Eye(int[] shape)
    {
        DenseTensor<float> tensor = new DenseTensor<float>(shape);
        tensor.Fill(0);
        int ndim = shape.Length;
        switch (ndim)
        {
            case 2:
            {
                int minShape = Math.Min(shape[0], shape[1]);

                for (int x = 0; x < minShape; x++)
                {
                    tensor[x, x] = 1;
                }

                break;
            }
            case 3:
            {
                int minShape = Math.Min(shape[1], shape[2]);
                Parallel.For(0, minShape, i =>
                {
                    for (int x = 0; x < shape[0]; x++)
                    {
                        tensor[x, i, i] = 1;
                    }
                });
                break;
            }
        }


        return tensor;
    }

    /// <summary>
    /// implement of numpy.r_
    /// </summary>
    /// <param name="array1">[663.5, 221.5, 1.063, 127]</param>
    /// <param name="array2">[0, 0, 0, 0]</param>
    /// <returns>[663.5, 221.5, 1.063, 127, 0, 0, 0, 0]</returns>
    public static DenseTensor<float> concatenateFlatten(DenseTensor<float> array1, DenseTensor<float> array2)
    {
        int length = (int)((int)array1.Length + array2.Length);
        int[] shape = new[] { length };
        DenseTensor<float> tensor = new DenseTensor<float>(shape);

        int count = 0;

        for (int i = 0; i < array1.Length; i++)
        {
            tensor[count] = array1[count];
            count++;
        }

        for (int i = 0; i < array2.Length; i++)
        {
            tensor[count] = array1[count];
            count++;
        }

        return tensor;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static DenseTensor<float> Square(DenseTensor<float> tensor)
    {
        int ndim = tensor.Rank;
        switch (ndim)
        {
            case 1:
            {
                for (var x = 0; x < tensor.Length; x++)
                {
                    tensor[x] *= tensor[x];
                }

                break;
            }
            case 2:
            {
                Parallel.For(0, tensor.Dimensions[0], h =>
                {
                    for (var y = 0; y < tensor.Dimensions[1]; y++)
                    {
                        tensor[h, y] *= tensor[h, y];
                    }
                });
                break;
            }
            case 3:
            {
                break;
            }
        }

        return tensor;
    }

    /// <summary>
    /// build new diagonal array from flat array
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static DenseTensor<float> Diag(DenseTensor<float> tensor)
    {
        int ndim = (int)tensor.Length;
        DenseTensor<float> resulTensor = new DenseTensor<float>(new ReadOnlySpan<int>(new[] { ndim, ndim }));
        switch (ndim)
        {
            case 1:
            {
                for (var x = 0; x < tensor.Length; x++)
                {
                    resulTensor[x, x] = tensor[x];
                }

                break;
            }
            case 2:
            {
                break;
            }
            case 3:
            {
                break;
            }
        }

        return tensor;
    }


    /// <summary>
    /// https://scicoding.com/how-to-calculate-cholesky-decomposition-in-python/
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static DenseTensor<float> CholeskyFactory(DenseTensor<float> tensor)
    {
        DenseTensor<float> resulTensor = new DenseTensor<float>(tensor.Dimensions);

        Parallel.For(0, 4, i =>
        {
            for (int j = 0; j < i + 1; j++)
            {
                float sum = 0;
                for (int k = 0; k < j + 1; k++)
                {
                    sum += resulTensor[i, k] * resulTensor[j, k];
                }

                if (i == j)
                {
                    resulTensor[i, j] = (float)Math.Sqrt(tensor[i, j] - sum);
                }
                else
                {
                    resulTensor[i, j] = 1.0f / resulTensor[j, j] * (tensor[i, j] - sum);
                }
            }
        });

        return resulTensor;
    }

    public static double[] Solve(double[][] L, double[] b)
    {
        int n = L.Length;
        double[] y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = b[i] / L[i][i];
            for (int j = 0; j < i; j++)
            {
                y[i] -= L[i][j] * y[j];
            }
        }

        return y;
    }

   
}