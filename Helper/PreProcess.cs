using Microsoft.ML.OnnxRuntime.Tensors;

namespace yolov7DotNet.Helper;

public abstract class PreProcess
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
        colors ??= new[] { 114f, 114f, 114f };

        var oriShape = image.Dimensions[1..]; //HW
        int[] newShape = new[] { 3, shapes[0], shapes[1] };

        DenseTensor<float> feed = new DenseTensor<float>(dimensions: newShape);
        // fill feed with special color
        Parallel.For(0, newShape[1], x =>
        {
            Parallel.For(0, newShape[2], y =>
            {
                feed[0, x, y] = colors[0];
                feed[1, x, y] = colors[1];
                feed[2, x, y] = colors[2];
            });
        });


        float r = Math.Min((float)shapes[0] / oriShape[0], (float)shapes[1] / oriShape[1]);
        if (!scaleUp)
        {
            r = Math.Min(r, 1.0f);
        }

        float[] ratio = new[] { r, r };
        int[] newUnPad = new[] { (int)Math.Round(oriShape[0] * r), (int)Math.Round(oriShape[1] * r) };
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
            image = ResizeLinear(image, newUnPad);
        }

        int left = (int)(Math.Round(dhdw[1] - 0.1));
        int top = (int)(Math.Round(dhdw[0] - 0.1));

        // implement of opencv copyMakeBorder
        Parallel.For(0, image.Dimensions[1], x =>
        {
            Parallel.For(0, image.Dimensions[2], y =>
            {
                feed[0, x + top, y + left] = image[0, x, y];
                feed[1, x + top, y + left] = image[1, x, y];
                feed[2, x + top, y + left] = image[2, x, y];
            });
        });
        
        return (feed, ratio, dhdw);
    }

    /// <summary>
    /// Convert ImageSharp to float32 DenseTensor
    /// </summary>
    /// <param name="image">ImageSharp</param>
    /// <returns>DenseTensor with shape 3, Height, Width</returns>
    public static DenseTensor<float> Image2DenseTensor(Image<Rgb24> image)
    {
        int[] shape = new[] { 3, image.Height, image.Width };

        DenseTensor<float> feed = new DenseTensor<float>(shape);

        Parallel.For(0, shape[1], y =>
        {
            Parallel.For(0, shape[2], x =>
            {
                feed[0, y, x] = image[x, y].R;
                feed[1, y, x] = image[x, y].G;
                feed[2, y, x] = image[x, y].B;
            });
        });

        return feed;
    }


    /// <summary>
    /// https://stackoverflow.com/a/69157357/22634632 image resize with numpy
    /// </summary>
    /// <param name="imageMatrix"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static DenseTensor<float> ResizeLinear(DenseTensor<float> imageMatrix, int[] shape)
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
            Parallel.For(0, shape[1], x =>
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
            });
        });
        return outputImage;
    }
}