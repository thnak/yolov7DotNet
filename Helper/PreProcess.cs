using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace yolov7DotNet.Helper;

public class PreProcess
{
    /// <summary>
    /// 
    /// </summary>
    /// <param name="image"></param>
    /// <param name="auto"></param>
    /// <param name="scaleFill"></param>
    /// <param name="scaleUp"></param>
    /// <param name="stride"></param>
    /// <param name="shapes">{Height, Width}</param>
    /// <param name="colors"></param>
    /// <returns></returns>
    public static (DenseTensor<float>, float[], float[]) LetterBox(DenseTensor<float> image,
        bool auto, bool scaleFill, bool scaleUp, int stride, int[] shapes, float[]? colors = null)
    {
        colors ??= new[] { 0.56470588235f, 0.56470588235f, 0.56470588235f };

        var oriShape = image.Dimensions; //CHW
        int[] newShape = new[] { oriShape[0], shapes[0], shapes[1] };

        DenseTensor<float> feed = new DenseTensor<float>(dimensions: newShape);

        float r = Math.Min((float)shapes[0] / oriShape[1], (float)shapes[1] / oriShape[2]);
        if (!scaleUp)
        {
            r = Math.Min(r, 1.0f);
        }

        float[] ratio = new[] { r, r };
        int[] newUnPad = new[] { (int)Math.Round(oriShape[1] * r), (int)Math.Round(oriShape[2] * r) };
        float[] dhdw = new[] { shapes[0] - newUnPad[0], (float)(shapes[1] - newUnPad[1]) };

        if (auto)
        {
            dhdw = new[] { dhdw[0] % stride, dhdw[1] % stride };
        }
        else if (scaleFill)
        {
            dhdw = new[] { 0f, 0f };
            newUnPad = new[] { shapes[0], shapes[1] };
            ratio = new[] { (float)shapes[0] / oriShape[0], (float)shapes[1] / oriShape[1] };
        }

        dhdw[0] /= 2;
        dhdw[1] /= 2;

        if (oriShape != newUnPad)
        {
            image = resize_linear(image, newUnPad);
        }

        // fill feed with special color
        Parallel.For(0, newShape[1], x =>
        {
            for (int y = 0; y < newShape[2]; y++)
            {
                feed[0, x, y] = colors[0];
                feed[1, x, y] = colors[1];
                feed[2, x, y] = colors[2];
            }
        });

        int left = (int)(Math.Round(dhdw[1] - 0.1));
        int right = (int)(Math.Round(dhdw[1] + 0.1));

        int top = (int)(Math.Round(dhdw[0] - 0.1));
        int bottom = (int)(Math.Round(dhdw[0] + 0.1));

        // implement of opencv copyMakeBorder
        Parallel.For(bottom, image.Dimensions[1], x =>
        {
            for (int y = left; y < image.Dimensions[2]; y++)
            {
                feed[0, x, y] = image[0, x - bottom, y - left];
                feed[1, x, y] = image[1, x - bottom, y - left];
                feed[2, x, y] = image[2, x - bottom, y - left];
            }
        });
        return (feed, ratio, dhdw);
    }

    /// <summary>
    /// Convert ImageSharp to float32 DenseTensor in range [0, 1]
    /// </summary>
    /// <param name="image">ImageSharp</param>
    /// <param name="maxShape"></param>
    /// <returns>DenseTensor with shape 3, Height, Width</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DenseTensor<float> Image2DenseTensor(Image<Rgb24> image, int? maxShape = null)
    {
        int[] shape = new[] { 3, image.Height, image.Width };
        int maxDim = Math.Max(image.Height, image.Width);
        if (maxShape is not null)
        {
            maxDim = Math.Max(maxDim, (int)maxShape);

            if (maxDim > maxShape)
            {
                float vectorMaxDim = (float)maxDim;
                float vectorMaxShape = (float)maxShape;
                var ratio = vectorMaxShape / vectorMaxDim;
                float width = image.Width;
                float height = image.Height;
                image.Mutate(context =>
                {
                    context.Resize(new ResizeOptions
                    {
                        Size = new Size((int)(width * ratio), (int)(height * ratio)),
                        Mode = ResizeMode.Crop
                    });
                });
            }
        }


        DenseTensor<float> feed = new DenseTensor<float>(shape);

        image.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < accessor.Height; y++)
            {
                var pixelSpan = accessor.GetRowSpan(y);
                for (var x = 0; x < accessor.Width; x++)
                {
                    feed[0, y, x] = pixelSpan[x].R / 255f;
                    feed[1, y, x] = pixelSpan[x].G / 255f;
                    feed[2, y, x] = pixelSpan[x].B / 255f;
                }
            }
        });

        return feed;
    }

    /// <summary>
    /// https://stackoverflow.com/a/69157357/22634632 image resize with numpy
    /// </summary>
    /// <param name="image"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public
        static DenseTensor<float> resize_linear(DenseTensor<float> imageMatrix, int[] shape)
    {
        int[] newShape = new[] { 3, shape[0], shape[1] };
        DenseTensor<float> outputImage = new DenseTensor<float>(newShape);

        var dim = imageMatrix.Dimensions.ToArray();

        var originalHeight = (float)dim[1]; //height
        var originalWidth = (float)dim[2]; //width

        var invScaleFactorY = originalHeight / shape[0];
        var invScaleFactorX = originalWidth / shape[1];

        Parallel.For(0, shape[0], y =>
        {
            for (int x = 0; x < shape[1]; x++)
            {
                var oldX = x * invScaleFactorX;
                var oldY = y * invScaleFactorY;
                var xFraction = oldX - (float)Math.Floor(oldX);
                var yFraction = oldY - (float)Math.Floor(oldY);
                // Sample four neighboring pixels:

                var leftUpperR = imageMatrix[0, (int)Math.Floor(oldY), (int)Math.Floor(oldX)];
                var leftUpperG = imageMatrix[1, (int)Math.Floor(oldY), (int)Math.Floor(oldX)];
                var leftUpperB = imageMatrix[2, (int)Math.Floor(oldY), (int)Math.Floor(oldX)];

                var rightUpperR = imageMatrix[0, (int)Math.Floor(oldY), (int)Math.Min(dim[2] - 1, Math.Ceiling(oldX))];
                var rightUpperG = imageMatrix[1, (int)Math.Floor(oldY), (int)Math.Min(dim[2] - 1, Math.Ceiling(oldX))];
                var rightUpperB = imageMatrix[2, (int)Math.Floor(oldY), (int)Math.Min(dim[2] - 1, Math.Ceiling(oldX))];

                var leftLowerR = imageMatrix[0, (int)Math.Min(dim[1] - 1, Math.Ceiling(oldY)), (int)Math.Floor(oldX)];
                var leftLowerG = imageMatrix[1, (int)Math.Min(dim[1] - 1, Math.Ceiling(oldY)), (int)Math.Floor(oldX)];
                var leftLowerB = imageMatrix[2, (int)Math.Min(dim[1] - 1, Math.Ceiling(oldY)), (int)Math.Floor(oldX)];

                var rightLowerR = imageMatrix[0, (int)Math.Min(dim[1] - 1, Math.Ceiling(oldY)), (int)Math.Min(dim[2] - 1, Math.Ceiling(oldX))];
                var rightLowerG = imageMatrix[1, (int)Math.Min(dim[1] - 1, Math.Ceiling(oldY)), (int)Math.Min(dim[2] - 1, Math.Ceiling(oldX))];
                var rightLowerB = imageMatrix[2, (int)Math.Min(dim[1] - 1, Math.Ceiling(oldY)), (int)Math.Min(dim[2] - 1, Math.Ceiling(oldX))];


                var blendTopR = (float)(rightUpperR * xFraction + leftUpperR * (1.0 - xFraction));
                var blendTopG = (float)(rightUpperG * xFraction + leftUpperG * (1.0 - xFraction));
                var blendTopB = (float)(rightUpperB * xFraction + leftUpperB * (1.0 - xFraction));

                var blendBottomR = (float)(rightLowerR * xFraction + leftLowerR * (1.0 - xFraction));
                var blendBottomG = (float)(rightLowerG * xFraction + leftLowerG * (1.0 - xFraction));
                var blendBottomB = (float)(rightLowerB * xFraction + leftLowerB * (1.0 - xFraction));


                var finalBlendR = (float)(blendTopR * yFraction + blendBottomR * (1.0 - yFraction));
                var finalBlendG = (float)(blendTopG * yFraction + blendBottomG * (1.0 - yFraction));
                var finalBlendB = (float)(blendTopB * yFraction + blendBottomB * (1.0 - yFraction));

                outputImage[0, y, x] = finalBlendR;
                outputImage[1, y, x] = finalBlendG;
                outputImage[2, y, x] = finalBlendB;
            }
        });
        return outputImage;
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
                    denseTensor[0, 1, i, x] = tensor[1, i, x];
                    denseTensor[0, 2, i, x] = tensor[2, i, x];
                    denseTensor[0, 3, i, x] = tensor[3, i, x];
                }
            });
        }

        return denseTensor;
    }
}