using System.Diagnostics;
using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.Fonts;
using yolov7DotNet.ModelsHelper;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;

namespace yolov7DotNet.Helper;

public class PostProcess
{
    /// <summary>
    /// 
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="dim"></param>
    /// <returns></returns>
    internal static DenseTensor<float> Xyxy2Xywh(DenseTensor<float> inputs, DenseTensor<float> dim)
    {
        DenseTensor<float> feed = new DenseTensor<float>(inputs.Dimensions);
        feed[0] = ((inputs[0] + inputs[2]) / 2) / dim[1];
        feed[1] = ((inputs[1] + inputs[3]) / 2) / dim[0];
        feed[2] = (inputs[2] - inputs[0]) / dim[1];
        feed[3] = (inputs[3] - inputs[1]) / dim[0];
        return feed;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="dim"></param>
    /// <returns></returns>
    internal static int[] Xyxy2Xywh(int[] inputs, int[] dim)
    {
        int[] feed = new int[inputs.Length];
        feed[0] = ((inputs[0] + inputs[2]) / 2) / dim[1];
        feed[1] = ((inputs[1] + inputs[3]) / 2) / dim[0];
        feed[2] = (inputs[2] - inputs[0]) / dim[1];
        feed[3] = (inputs[3] - inputs[1]) / dim[0];
        return feed;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="modelPreds">Tensor with shape (batch x 7)</param>
    /// <param name="imageShape">the original shape of image x</param>
    /// <param name="dwdhs"></param>
    /// <param name="ratios"></param>
    /// <param name="names"></param>
    /// <param name="confThres"></param>
    /// <returns></returns>
    internal static List<Models.Yolov7Predict> End2End(DenseTensor<float> modelPreds, List<int[]> imageShape,
        List<float[]> dwdhs, List<float[]> ratios, string[] names, float confThres = 0.2f)
    {
        List<Models.Yolov7Predict> yolov7Predicts = new();
        int firstDim = modelPreds.Dimensions[0];

        Parallel.For(0, firstDim, x =>
        {
            var score = modelPreds[x, 6];
            if (score < confThres)
            {
                return;
            }

            var batch_id = (int)modelPreds[x, 0];
            var x0 = modelPreds[x, 1];
            var y0 = modelPreds[x, 2];
            var x1 = modelPreds[x, 3];
            var y1 = modelPreds[x, 4];
            var cls_id = (int)modelPreds[x, 5];

            float[] floatBox = new[] { x0, y0, x1, y1 };
            float[] doubleDwdhs = new[] { dwdhs[batch_id][0], dwdhs[batch_id][1], dwdhs[batch_id][0], dwdhs[batch_id][1] };
            float[] doubleratio = new[] { ratios[batch_id][0], ratios[batch_id][1], ratios[batch_id][0], ratios[batch_id][1] };

            floatBox[0] -= doubleDwdhs[0];
            floatBox[1] /= doubleratio[1];
            floatBox[2] -= doubleDwdhs[2];
            floatBox[3] /= doubleratio[3];

            int[] box = new[] { (int)Math.Round(floatBox[0]), (int)Math.Round(floatBox[1]), (int)Math.Round(floatBox[2]), (int)Math.Round(floatBox[3]) };
            yolov7Predicts.Add(new Models.Yolov7Predict()
            {
                BatchId = batch_id,
                Bbox = Xyxy2Xywh(box, imageShape[batch_id]),
                Box = box,
                Score = score,
                ClassIdx = cls_id,
                ClassName = names.ElementAt(cls_id)
            });
        });

        return yolov7Predicts;
    }

    public static List<Image<Rgb24>> DrawBox(List<Image<Rgb24>> listTensor, List<Models.Yolov7Predict> listPredict)
    {
        var fo = SystemFonts.Families.First();
        Font font = new Font(fo, 16);
        List<Image<Rgb24>> results = new List<Image<Rgb24>>();
        Parallel.For(0, listPredict.Count, i =>
        {
            Models.Yolov7Predict yolov7Predict = listPredict[i];
            Image<Rgb24> image = listTensor[0];
            var minShape = Math.Min(image.Height, image.Width);
            minShape /= 480;
            minShape = Math.Min(minShape, 1);
            var (x, y) = (yolov7Predict.Bbox[0], yolov7Predict.Bbox[1] - 23);
            image.Mutate(a =>
            {
                a.Draw(Id2Colors.Int2Color[Id2Colors.Int2Color.Count - 1 - yolov7Predict.ClassIdx], minShape, new RectangleF(yolov7Predict.Bbox[0], yolov7Predict.Bbox[1], yolov7Predict.Bbox[2], yolov7Predict.Bbox[3]));
                a.DrawText(new DrawingOptions(), $"{yolov7Predict.ClassName} {Math.Round(yolov7Predict.Score, 2)}", font, Helper.Id2Colors.Int2Color[yolov7Predict.ClassIdx], new PointF(x, y));
            });
            results.Add(image);
        });
        return results;
    }
}