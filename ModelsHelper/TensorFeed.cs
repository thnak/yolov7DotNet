using Microsoft.ML.OnnxRuntime.Tensors;
using Numpy;
using OpenCvSharp;
using yolov7DotNet.Helper;
using yolov7DotNet.Operators;

namespace yolov7DotNet.ModelsHelper;

public class TensorFeed
{
    private List<DenseTensor<float>> FeedQueue { get; set; } = new List<DenseTensor<float>>();
    private List<NDarray> NDarrays = new List<NDarray>();
    private NDarray ImageBatched { get; set; }
    private List<int[]> ImageShape { get; set; } = new List<int[]>();
    private List<float[]> Ratios { get; set; } = new List<float[]>();
    private List<float[]> Dwdhs { get; set; } = new List<float[]>();
    private int[] OutPutShape { get; set; }
    private int Stride { get; set; }

    private int Length => FeedQueue.Count;

    public TensorFeed(int[] outPutShape, int stride)
    {
        OutPutShape = outPutShape;
        Stride = stride;
        ImageBatched = np.zeros(new[] { 32, 3, 640, 640 });
    }

    public void SetTensor(Mat image)
    {
        Thread thread = new Thread(() =>
        {
            Cv2.CvtColor(image, image, ColorConversionCodes.BGR2RGB);
            var lettered = PreProcess.LetterBox(image, false, false, true, Stride, OutPutShape);
            lettered.Item1.Reshape(1).GetArray(out byte[] image2);
            NDarray nDarray = np.array(image2).reshape(OutPutShape[0], OutPutShape[1], 3).astype(np.float32);
            nDarray = nDarray.transpose(new[] { 2, 0, 1 });
            nDarray /= 255f;
            nDarray = nDarray.expand_dims(0);
            NDarrays.Add(nDarray);
            Dwdhs.Add(lettered.Item3);
            Ratios.Add(lettered.Item2);
            ImageShape.Add(new[] { image.Height, image.Width });
            ImageBatched = np.concatenate(NDarrays.ToArray());
        });
        thread.Start();
        thread.Join();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <returns></returns>
    public (DenseTensor<float>, List<float[]>, List<float[]>, List<int[]>) GetTensor()
    {
        try
        {
            var shape = ImageBatched.shape.Dimensions;
            DenseTensor<float> tensor = new DenseTensor<float>(shape);
            Parallel.For(0, shape[2], h =>
            {
                for (var b = 0; b < shape[0]; b++)
                {
                    for (var w = 0; w < shape[3]; w++)
                    {
                        tensor[b, 0, h, w] = ImageBatched[b, 0, h, w].item<float>();
                        tensor[b, 1, h, w] = ImageBatched[b, 1, h, w].item<float>();
                        tensor[b, 2, h, w] = ImageBatched[b, 2, h, w].item<float>();
                    }
                }
            });
            return (tensor, Dwdhs, Ratios, ImageShape);
        }
        finally
        {
            NDarrays.Clear();
        }
    }

    public async Task SetTensorAsync(DenseTensor<float> tensor)
    {
        var imageShape = tensor.Dimensions[1..].ToArray();
        var lettered = await Task.FromResult(PreProcess.LetterBox(tensor, false, false, true, Stride, OutPutShape));
        var feedTensor = await Task.FromResult(Ops.ExpandDim(Ops.Div(lettered.Item1, 255f)));
        FeedQueue.Add(feedTensor);
        Dwdhs.Add(lettered.Item3);
        Ratios.Add(lettered.Item2);
        ImageShape.Add(imageShape);
    }

    public void SetTensor(DenseTensor<float> tensor)
    {
        var imageShape = tensor.Dimensions[1..].ToArray();
        var lettered = PreProcess.LetterBox(tensor, false, false, true, Stride, OutPutShape);
        var feedTensor = Ops.ExpandDim(Ops.Div(lettered.Item1, 255f));
        FeedQueue.Add(feedTensor);
        Dwdhs.Add(lettered.Item3);
        Ratios.Add(lettered.Item2);
        ImageShape.Add(imageShape);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <returns></returns>
    public async Task<(DenseTensor<float>, List<float[]>, List<float[]>, List<int[]>)> GetTensorAsync()
    {
        DenseTensor<float> tensor = await Task.FromResult(Ops.Concat(FeedQueue));
        return (tensor, Dwdhs, Ratios, ImageShape);
    }
}