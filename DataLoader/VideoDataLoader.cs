using System.Collections;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace yolov7DotNet.DataLoader;

public class VideoDataLoader
{
    private VideoCapture VideoCapture { get; set; }
    private int BatchSize { get; set; } = 1;
    private int Height { get; set; }
    private int Width { get; set; }
    private float Fps { get; set; }
    private int[] Dim { get; set; }

    public VideoDataLoader(int batchSize,int? cameraIdx, string? fileString, string? autoInt)
    {
        if (cameraIdx is not null) {VideoCapture = VideoCapture.FromCamera((int)cameraIdx);}
        if(fileString != null) {VideoCapture = VideoCapture.FromFile(fileString);}

        if (autoInt != null)
        {
            VideoCapture = new VideoCapture(autoInt);
        }

        Height = VideoCapture.FrameHeight;
        Width = VideoCapture.FrameWidth;
        Fps = (float)VideoCapture.Fps;
        BatchSize = batchSize;
        Dim = new[] { batchSize, 3, Height, Width };
    }
    
    public IEnumerator<DenseTensor<float>> GetEnumerator()
    {
        DenseTensor<float> tensor;
        List<DenseTensor<float>> listTensor = new List<DenseTensor<float>>();
        while (VideoCapture.IsOpened())
        {
            Mat frame = VideoCapture.RetrieveMat();
            Cv2.CvtColor(frame, frame, ColorConversionCodes.BGR2RGB);
            var read = Helper.PreProcess.Stream2Tensor(frame.ToMemoryStream());
            if (listTensor.Count() < BatchSize)
            {
                listTensor.Add(read);
            }
            else
            {
                tensor = Operators.Ops.Concat(listTensor);
                listTensor = new List<DenseTensor<float>>();
                yield return tensor;
            }
        }
    }
}