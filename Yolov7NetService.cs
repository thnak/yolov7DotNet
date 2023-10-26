using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.JSInterop;
using Microsoft.ML.OnnxRuntime;
using yolov7DotNet.Helper;

namespace yolov7DotNet;

/// <summary>
/// 
/// </summary>
public class Yolov7NetService
{
    /// <summary>
    /// model weight include in this project
    /// Default is tiny model
    /// </summary>
    public enum ModelWeights
    {
        /// <summary>
        /// https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
        /// </summary>
        Yolov7Tiny,

        /// <summary>
        /// https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
        /// </summary>
        Yolov7,

        /// <summary>
        /// https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
        /// </summary>
        DefaultModel
    }


    private interface IYolov7
    {
        Task<List<Models.Yolov7Predict>> InferenceAsync(MemoryStream memoryStream);
        Task<List<Models.Yolov7Predict>> InferenceAsync(Image<Rgb24> image);
        Task<List<Models.Yolov7Predict>> InferenceAsync(string fileDir);
        Task<List<Models.Yolov7Predict>> InferenceAsync(Stream stream);
    }

    /// <summary>
    /// 
    /// </summary>
    public class Yolov7 : IYolov7, IDisposable
    {
        private readonly string _prefix = Properties.Resources.prefix;
        private readonly SessionOptions _sessionOptions;
        private readonly InferenceSession _session;
        private readonly RunOptions _runOptions;
        private readonly HashSet<string> _categories;
        private readonly IEnumerable<string> _inputNames;
        private readonly IReadOnlyList<string> _outputNames;
        private int Stride { get; set; }
        private bool _disposed;
        private IMemoryCache MemoryCache { get; }

        private readonly int[] _inputShape;
        private readonly IJSRuntime? _jsRuntime;

        /// <summary>
        /// init new prediction session
        /// </summary>
        /// <param name="jsRuntime">optional for blazor</param>
        /// <param name="weight">model in this project</param>
        /// <param name="byteWeight">you can use your own model with this argument, make sure it include NMS and output shape is tensor: float32[Concatoutput_dim_0,7]</param>
        /// <exception cref="Exception">can not init for some reason</exception>
        public Yolov7(IJSRuntime? jsRuntime = null, ModelWeights? weight = null,
            byte[]? byteWeight = null)
        {
            if (jsRuntime != null)
            {
                _jsRuntime = jsRuntime;
            }

            _sessionOptions = new SessionOptions();
            _sessionOptions.EnableMemoryPattern = true;
            _sessionOptions.EnableCpuMemArena = true;
            _sessionOptions.EnableProfiling = false;
            _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            var availableProvider = OrtEnv.Instance().GetAvailableProviders()[0];


            switch (availableProvider)
            {
                case "CUDAExecutionProvider":
                {
                    OrtCUDAProviderOptions providerOptions = new OrtCUDAProviderOptions();
                    var providerOptionsDict = new Dictionary<string, string>
                    {
                        ["cudnn_conv_use_max_workspace"] = "1",
                        ["device_id"] = "0"
                    };
                    providerOptions.UpdateOptions(providerOptionsDict);
                    _sessionOptions.AppendExecutionProvider_CUDA(providerOptions);
                    JsLogger($"[{_prefix}][INIT][CUDAExecutionProvider]");
                    break;
                }
                case "TensorrtExecutionProvider":
                {
                    OrtTensorRTProviderOptions provider = new OrtTensorRTProviderOptions();
                    string result = Path.GetTempPath();
                    string temPath = Path.Combine(result, $"_yolov7NetService_{weight}.engine");
                    var providerOptionsDict = new Dictionary<string, string>
                    {
                        ["cudnn_conv_use_max_workspace"] = "1",
                        ["device_id"] = "0",
                        ["ORT_TENSORRT_FP16_ENABLE"] = "true",
                        ["ORT_TENSORRT_LAYER_NORM_FP32_FALLBACK"] = "true",
                        ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "true",
                        ["ORT_TENSORRT_CACHE_PATH"] = $"{temPath}"
                    };
                    provider.UpdateOptions(providerOptionsDict);
                    _sessionOptions.AppendExecutionProvider_Tensorrt(provider);
                    JsLogger($"[{_prefix}][INIT][TensorrtExecutionProvider]");

                    break;
                }
                case "DNNLExecutionProvider":
                {
                    _sessionOptions.AppendExecutionProvider_Dnnl();
                    break;
                }
                case "OpenVINOExecutionProvider":
                {
                    _sessionOptions.AppendExecutionProvider_OpenVINO();
                    _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
                    JsLogger($"[{_prefix}][INIT][OpenVINOExecutionProvider]");

                    break;
                }
                case "DmlExecutionProvider":
                {
                    _sessionOptions.EnableMemoryPattern = false;
                    _sessionOptions.AppendExecutionProvider_DML();
                    JsLogger($"[{_prefix}][INIT][DmlExecutionProvider]");

                    break;
                }
                case "ROCMExecutionProvider":
                {
                    OrtROCMProviderOptions provider = new();
                    var providerOptionsDict = new Dictionary<string, string>
                    {
                        ["device_id"] = "0",
                        ["cudnn_conv_use_max_workspace"] = "1"
                    };
                    provider.UpdateOptions(providerOptionsDict);
                    _sessionOptions.AppendExecutionProvider_ROCm(provider);
                    JsLogger($"[{_prefix}][INIT][ROCMExecutionProvider]");

                    break;
                }
            }

            var prepackedWeightsContainer = new PrePackedWeightsContainer();
            _runOptions = new RunOptions();

            if (weight is not null)
            {
                switch (weight)
                {
                    case ModelWeights.Yolov7:
                    {
                        JsLogger($"[{_prefix}][INIT][ModelWeights][yolov7]");
                        break;
                    }
                    case ModelWeights.Yolov7Tiny:
                    {
                        _session = new InferenceSession(Properties.Resources.yolov7_tiny, _sessionOptions, prepackedWeightsContainer);
                        JsLogger($"[{_prefix}][INIT][ModelWeights][yolov7_tiny]");
                        break;
                    }
                    default:
                    {
                        _session = new InferenceSession(Properties.Resources.yolov7_tiny, _sessionOptions, prepackedWeightsContainer);
                        JsLogger($"[{_prefix}][INIT][ModelWeights][yolov7_tiny]");
                        break;
                    }
                }
            }
            else if (byteWeight is not null)
            {
                _session = new InferenceSession(byteWeight, _sessionOptions, prepackedWeightsContainer);
                JsLogger($"[{_prefix}][INIT][ModelWeights][byte model]");
            }

            else
            {
                _session = new InferenceSession(Properties.Resources.yolov7_tiny, _sessionOptions, prepackedWeightsContainer);
                JsLogger($"[{_prefix}][INIT][ModelWeights][yolov7_tiny]");
            }

            var metadata = _session?.ModelMetadata;
            var customMetadata = metadata?.CustomMetadataMap;
            if (customMetadata != null && customMetadata.TryGetValue("names", out var categories))
            {
                var content = JsonSerializer.Deserialize<List<string>>(categories);
                if (content != null) _categories = new HashSet<string>(content);
                else
                {
                    JsLogger($"[{_prefix}][Init][ERROR] not found categories in model metadata, creating name with syntax Named[?]");
                    _categories = new HashSet<string>();
                    for (var i = 0; i < 10000; i++)
                    {
                        _categories.Add($"Named[{i}]");
                    }
                }
            }
            else
            {
                JsLogger($"[{_prefix}][Init][ERROR] not found categories in model metadata, creating name with syntax Named[?]");
                _categories = new HashSet<string>();
                for (var i = 0; i < 10000; i++)
                {
                    _categories.Add($"Named[{i}]");
                }
            }

            if (customMetadata != null && customMetadata.TryGetValue("stride", out string? stride))
            {
                var content = JsonSerializer.Deserialize<List<int>>(stride);
                if (content != null) Stride = content.Last();
            }
            else
            {
                Stride = 32;
                JsLogger($"[{_prefix}][Init][ERROR][STRIDE] not found stride, set to default 32");
            }

            if (_session != null)
            {
                _inputNames = _session.InputNames;
                _outputNames = _session.OutputNames;
                _inputShape = _session.InputMetadata.First().Value.Dimensions;
                MemoryCache = new MemoryCache(new MemoryCacheOptions());
                MemoryCache.Set("_runOptions", _runOptions);
                MemoryCache.Set("_sessionOptions", _sessionOptions);
                MemoryCache.Set("_session", _session);
                if (_jsRuntime is not null) MemoryCache.Set("_jsRuntime", _jsRuntime);
            }
            else
            {
                throw new Exception($"[{_prefix}][Init][ERROR] could not init");
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="byteArray"></param>
        /// <returns></returns>
        public async Task<List<Models.Yolov7Predict>> InferenceAsync(byte[] byteArray)
        {
            Image<Rgb24> image = Image.Load<Rgb24>(byteArray);
            return await InferenceAsync(image);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="memoryStream"></param>
        /// <returns></returns>
        public async Task<List<Models.Yolov7Predict>> InferenceAsync(MemoryStream memoryStream)
        {
            Image<Rgb24> image = Image.Load<Rgb24>(memoryStream);
            return await InferenceAsync(image);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="fileDir"></param>
        /// <returns></returns>
        public async Task<List<Models.Yolov7Predict>> InferenceAsync(string fileDir)
        {
            Image<Rgb24> image = Image.Load<Rgb24>(fileDir);
            return await InferenceAsync(image);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="stream"></param>
        /// <returns></returns>
        public async Task<List<Models.Yolov7Predict>> InferenceAsync(Stream stream)
        {
            Image<Rgb24> image = Image.Load<Rgb24>(stream);
            return await InferenceAsync(image);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        public async Task<List<Models.Yolov7Predict>> InferenceAsync(Image<Rgb24> image)
        {
            var stopwatch = new Stopwatch();
            stopwatch.Start();

            var tensorFeed = PreProcess.Image2DenseTensor(image, _inputShape.Last());
            var imageShape = tensorFeed.Dimensions[1..].ToArray();
            var lettered = PreProcess.LetterBox(tensorFeed, true, false, true, Stride, new[] { _inputShape[2], _inputShape[3] });

            var letteredItem1Dim = lettered.Item1.Dimensions.ToArray();
            long[] newDim = new[] { 1L, letteredItem1Dim[0], letteredItem1Dim[1], letteredItem1Dim[2] };

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, PreProcess.ExpandDim(lettered.Item1).Buffer, newDim);
            var inputs = new Dictionary<string, OrtValue> { { _inputNames.First(), inputOrtValue } };

            using var fromResult = await Task.FromResult(_session.Run(_runOptions, inputs, _outputNames));
            float[] resultArrays = fromResult.First().Value.GetTensorDataAsSpan<float>().ToArray();
            Models.Predictions predictions = new Models.Predictions(resultArrays, _categories.ToArray(), new List<float[]>() { lettered.Item3 }, new List<float[]>(){lettered.Item2}, new List<int[]>() { imageShape });
            return predictions.GetDetect();
        }

        /// <summary>
        /// 
        /// </summary>
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="disposing"></param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed) return;
            if (disposing)
            {
                MemoryCache.Dispose();
                _sessionOptions.Dispose();
                _session.Dispose();
                _runOptions.Dispose();
                JsLogger("[ImageClassifyService][Dispose] disposed");
            }

            _disposed = true;
        }

        private void JsLogger(string message)
        {
            if (_jsRuntime is not null)
            {
                _jsRuntime.InvokeVoidAsync("console.log", message);
            }
        }
    }
}