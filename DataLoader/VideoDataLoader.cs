using OpenCvSharp;
using yolov7DotNet.ModelsHelper;

namespace yolov7DotNet.DataLoader;

public class VideoDataLoader
{
    private VideoCapture VideoCapture { get; set; }
    private int BatchSize { get; set; } = 1;
    private int Height { get; set; }
    private int Width { get; set; }
    private float Fps { get; set; }
    
    private int[] OutputShape { get; set; }
    private int Stride { get; set; }

    public VideoDataLoader(int batchSize,int? cameraIdx, string? fileString, string? autoInt, int[] outputShape, int stride)
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
        OutputShape = outputShape;
        Stride = stride;
    }

   
    public IEnumerator<TensorFeed> GetEnumerator()
    {
        if (BatchSize == 1)
        {
            while (VideoCapture.IsOpened())
            {
                Mat frame = VideoCapture.RetrieveMat();
                if (frame == null)
                {
                    break;
                }
                
                Cv2.CvtColor(frame, frame, ColorConversionCodes.BGR2RGB);
                var read = Helper.PreProcess.Stream2Tensor(frame.ToMemoryStream());
                TensorFeed tensorFeed = new TensorFeed(OutputShape, Stride);
                tensorFeed.SetTensor(read);
                yield return tensorFeed;
            }
        }
        else
        {
            int counting = 0;
            TensorFeed tensorFeed = new TensorFeed(OutputShape, Stride);
            while (VideoCapture.IsOpened())
            {
                Mat frame = VideoCapture.RetrieveMat();
                if (frame == null)
                {
                    break;
                }
                 
                Cv2.CvtColor(frame, frame, ColorConversionCodes.BGR2RGB);
                var read = Helper.PreProcess.Stream2Tensor(frame.ToMemoryStream());
                if (counting < BatchSize)
                {
                    tensorFeed.SetTensor(read);
                    counting++;
                }
                else
                {
                    try
                    {
                        yield return tensorFeed;
                    }
                    finally
                    {
                        tensorFeed = new TensorFeed(OutputShape, Stride);
                        counting = 0;
                    }
                }
            }
        }
    }
}