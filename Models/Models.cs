using yolov7DotNet.Helper;

namespace yolov7DotNet;

public class Models
{
    /// <summary>
    /// Base output from yolov7 non-maxsuppression
    /// </summary>
    public class Yolov7Predict
    {
        /// <summary>
        /// index of batch
        /// </summary>
        public int BatchId { get; set; }

        /// <summary>
        /// bounding box
        /// </summary>
        public int[]? Bbox { get; set; }

        /// <summary>
        /// box
        /// </summary>
        public int[]? Box { get; set; }

        /// <summary>
        /// category index
        /// </summary>
        public int ClassIdx { get; set; }

        /// <summary>
        /// category
        /// </summary>
        public string? ClassName { get; set; }

        /// <summary>
        /// confident score
        /// </summary>
        public float Score { get; set; }
    }

    /// <summary>
    /// simple predict object
    /// </summary>
    public class Predictions
    {
        private float[] PredictionArrays { get; set; }
        private string[] Categories { get; set; }
        private List<float[]> Dwdhs { get; set; }
        private List<int[]> ImageShapes { get; set; }
        private List<float[]> Ratios { get; set; }

        /// <summary>
        /// init an object that hold all prediction at one
        /// </summary>
        /// <param name="predictionArrayResults"></param>
        /// <param name="categories"></param>
        /// <param name="dhdws"></param>
        /// <param name="ratios"></param>
        /// <param name="imageShapes"></param>
        public Predictions(float[] predictionArrayResults, string[] categories, List<float[]> dhdws, List<float[]> ratios, List<int[]> imageShapes)
        {
            PredictionArrays = predictionArrayResults;
            Categories = categories;
            Dwdhs = dhdws;
            ImageShapes = imageShapes;
            Ratios = ratios;
        }

        /// <summary>
        /// get all prediction
        /// </summary>
        /// <returns></returns>
        public List<Yolov7Predict> GetDetect()
        {
            List<Yolov7Predict> yolov7Predicts = new List<Yolov7Predict>();

            int length = PredictionArrays.Length;
            if (length > 0)
            {
                length /= 7;
                Parallel.For(0, length, i =>
                {
                    int end = i + 7;
                    float[] slice = PredictionArrays[i..end];
                    int clsIdx = (int)slice[5];
                    int batchId = (int)slice[0];
                    float[] boxArray = new[] { slice[1], slice[2], slice[3], slice[4] };
                    float[] doubleRatios = new[] { Ratios[batchId][0], Ratios[batchId][1], Ratios[batchId][0], Ratios[batchId][1] };
                    float[] doubleDwDhs = new[] { Dwdhs[batchId][0], Dwdhs[batchId][1], Dwdhs[batchId][0], Dwdhs[batchId][1] };

                    boxArray[0] -= doubleDwDhs[0];
                    boxArray[1] /= doubleRatios[1];
                    boxArray[2] -= doubleDwDhs[2];
                    boxArray[3] /= doubleRatios[3];

                    int[] box = new[] { (int)Math.Round(boxArray[0]), (int)Math.Round(boxArray[1]), (int)Math.Round(boxArray[2]), (int)Math.Round(boxArray[3]) };

                    yolov7Predicts.Add(new Yolov7Predict()
                    {
                        BatchId = (int)slice[0],
                        ClassIdx = clsIdx,
                        Score = slice[6],
                        ClassName = Categories[clsIdx],
                        Box = box,
                        Bbox = Xyxy2Xywh(box, ImageShapes[batchId])
                    });
                });
            }

            return yolov7Predicts;
        }
    }

    /// <summary>
    /// convert xyxy to xywh 
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="dim"></param>
    /// <returns></returns>
    private static int[] Xyxy2Xywh(IReadOnlyList<int> inputs, IReadOnlyList<int> dim)
    {
        var feed = new int[inputs.Count];
        feed[0] = ((inputs[0] + inputs[2]) / 2) / dim[1];
        feed[1] = ((inputs[1] + inputs[3]) / 2) / dim[0];
        feed[2] = (inputs[2] - inputs[0]) / dim[1];
        feed[3] = (inputs[3] - inputs[1]) / dim[0];
        return feed;
    }
}