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

    public static List<Image<Rgb24>> DrawBox(List<Image<Rgb24>> listTensor, List<Models.Yolov7Predict> listPredict)
    {
        var fo = SystemFonts.Families.First();
        Font font = new Font(fo, 16);
        List<Image<Rgb24>> results = new List<Image<Rgb24>>();
        Parallel.For(0, listPredict.Count, i =>
        {
            Models.Yolov7Predict yolov7Predict = listPredict[i];
            Image<Rgb24> image = listTensor[yolov7Predict.BatchId];
            var minShape = Math.Min(image.Height, image.Width);
            minShape /= 480;
            minShape = Math.Min(minShape, 1);
            var (x, y) = (yolov7Predict.Bbox[0], yolov7Predict.Bbox[1] - 23);
            image.Mutate(a =>
            {
                a.Draw(Id2Colors.Int2Color[Id2Colors.Int2Color.Count - 1 - yolov7Predict.ClassIdx], minShape, new RectangleF(yolov7Predict.Bbox[0], yolov7Predict.Bbox[1], yolov7Predict.Bbox[2], yolov7Predict.Bbox[3]));
                a.DrawText(new DrawingOptions(), $"{yolov7Predict.ClassName} {Math.Round(yolov7Predict.Score, 2)}", font, Id2Colors.Int2Color[yolov7Predict.ClassIdx], new PointF(x, y));
            });
            results.Add(image);
        });
        return results;
    }
}