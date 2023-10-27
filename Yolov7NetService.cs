using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;
using Microsoft.JSInterop;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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

    public enum ExecutionProvider
    {
        /// <summary>
        /// 
        /// </summary>
        CPU,

        /// <summary>
        /// 
        /// </summary>
        CUDA,

        /// <summary>
        /// 
        /// </summary>
        TensorRT,
        DML,
        OpenCL,
        ROCm,
        OpenVINO,
        oneDNN,
        QNN,
        NNAPI,
        CoreML,
        XNNPACK,
        Azure,
        DNNL
    }

    private interface IYolov7
    {
        Task<List<Models.Yolov7Predict>> InferenceAsync(MemoryStream memoryStream);
        Task<List<Models.Yolov7Predict>> InferenceAsync(Image<Rgb24> image);
        Task<List<Models.Yolov7Predict>> InferenceAsync(string fileDir);
        Task<List<Models.Yolov7Predict>> InferenceAsync(Stream stream);
        Task<List<Models.Yolov7Predict>> InferenceAsync(DenseTensor<float> tensor);
        List<string> GetAvailableProviders();
        void SetExcutionProvider(ExecutionProvider ex);
    }

    /// <summary>
    /// 
    /// </summary>
    public class Yolov7 : IYolov7, IDisposable
    {
        private readonly string _prefix = Properties.Resources.prefix;
        private SessionOptions _sessionOptions;
        private readonly InferenceSession _session;
        private readonly RunOptions _runOptions;
        private readonly List<string> _categories;
        private readonly IEnumerable<string> _inputNames;
        private readonly IReadOnlyList<string> _outputNames;
        private int Stride { get; set; }
        private bool _disposed;
        private IMemoryCache MemoryCache { get; }

        private readonly int[] _inputShape;
        private readonly IJSRuntime? _jsRuntime;
        private readonly ILogger<Yolov7>? _logger;

        public Yolov7() : this(weight: ModelWeights.Yolov7Tiny, jsRuntime: null, byteWeight: null)
        {
        }

        public Yolov7(ModelWeights modelWeights) : this(weight: modelWeights, jsRuntime: null, byteWeight: null)
        {
        }

        public Yolov7(ModelWeights modelWeights, JSRuntime jsRuntime) : this(weight: modelWeights, jsRuntime: jsRuntime, byteWeight: null)
        {
        }

        public Yolov7(ModelWeights modelWeights, ILogger<Yolov7> logger) : this(weight: modelWeights, jsRuntime: null, byteWeight: null, logger: logger)
        {
        }

        //
        public Yolov7(byte[] modelWeights) : this(byteWeight: modelWeights, jsRuntime: null, logger: null)
        {
        }

        public Yolov7(byte[] modelWeights, JSRuntime jsRuntime) : this(byteWeight: modelWeights, jsRuntime: jsRuntime, logger: null)
        {
        }

        public Yolov7(byte[] modelWeights, ILogger<Yolov7> logger) : this(byteWeight: modelWeights, jsRuntime: null, logger: logger)
        {
        }

        //
        public Yolov7(IJSRuntime? jsRuntime = null, ModelWeights? weight = null, byte[]? byteWeight = null) : this(jsRuntime: jsRuntime, logger: null, weight: weight, byteWeight: byteWeight)
        {
        }


        public Yolov7(IJSRuntime? jsRuntime = null, ILogger<Yolov7>? logger = null, ModelWeights? weight = null, byte[]? byteWeight = null)
        {
            if (jsRuntime != null)
            {
                _jsRuntime = jsRuntime;
            }

            if (logger != null)
            {
                _logger = logger;
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
                    SetExcutionProvider(ExecutionProvider.CUDA);
                    TheLogger($"[{_prefix}][INIT][CUDAExecutionProvider]");
                    break;
                }
                case "TensorrtExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.TensorRT);
                    TheLogger($"[{_prefix}][INIT][TensorrtExecutionProvider]");

                    break;
                }
                case "DNNLExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.DNNL);
                    TheLogger($"[{_prefix}][INIT][DNNLExecutionProvider]");
                    break;
                }
                case "OpenVINOExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.OpenVINO);
                    TheLogger($"[{_prefix}][INIT][OpenVINOExecutionProvider]");

                    break;
                }
                case "DmlExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.DML);
                    TheLogger($"[{_prefix}][INIT][DmlExecutionProvider]");
                    break;
                }
                case "ROCMExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.ROCm);
                    TheLogger($"[{_prefix}][INIT][ROCMExecutionProvider]");

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
                        TheLogger($"[{_prefix}][INIT][ModelWeights][yolov7]");
                        break;
                    }
                    case ModelWeights.Yolov7Tiny:
                    {
                        _session = new InferenceSession(Properties.Resources.yolov7_tiny, _sessionOptions, prepackedWeightsContainer);
                        TheLogger($"[{_prefix}][INIT][ModelWeights][yolov7_tiny]");
                        break;
                    }
                    default:
                    {
                        _session = new InferenceSession(Properties.Resources.yolov7_tiny, _sessionOptions, prepackedWeightsContainer);
                        TheLogger($"[{_prefix}][INIT][ModelWeights][yolov7_tiny]");
                        break;
                    }
                }
            }
            else if (byteWeight is not null)
            {
                _session = new InferenceSession(byteWeight, _sessionOptions, prepackedWeightsContainer);
                TheLogger($"[{_prefix}][INIT][ModelWeights][byte model]");
            }

            else
            {
                _session = new InferenceSession(Properties.Resources.yolov7_tiny, _sessionOptions, prepackedWeightsContainer);
                TheLogger($"[{_prefix}][INIT][ModelWeights][yolov7_tiny]");
            }

            var metadata = _session?.ModelMetadata;
            var customMetadata = metadata?.CustomMetadataMap;
            Debug.Assert(customMetadata != null, nameof(customMetadata) + " != null");
            if (customMetadata.TryGetValue("names", out var categories))
            {
                var content = JsonSerializer.Deserialize<List<string>>(categories);
                if (content != null) _categories = content;
                else
                {
                    TheLogger($"[{_prefix}][Init][ERROR] not found categories in model metadata, creating name with syntax Named[?]");
                    _categories = new List<string>();
                    for (var i = 0; i < 10000; i++)
                    {
                        _categories.Add($"Named[ {i} ]");
                    }
                }
            }
            else
            {
                TheLogger($"[{_prefix}][Init][ERROR] not found categories in model metadata, creating name with syntax Named[?]");
                _categories = new List<string>();
                for (var i = 0; i < 10000; i++)
                {
                    _categories.Add($"Named[ {i} ]");
                }
            }

            if (customMetadata.TryGetValue("stride", out string? stride))
            {
                var content = JsonSerializer.Deserialize<List<float>>(stride);
                if (content != null) Stride = (int)content.Last();
            }
            else
            {
                Stride = 32;
                TheLogger($"[{_prefix}][Init][ERROR][STRIDE] not found stride, set to default 32");
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
            using var image = Image.Load<Rgb24>(byteArray);
            return await InferenceAsync(image);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="memoryStream"></param>
        /// <returns></returns>
        public async Task<List<Models.Yolov7Predict>> InferenceAsync(MemoryStream memoryStream)
        {
            using var image = Image.Load<Rgb24>(memoryStream);
            return await InferenceAsync(image);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="fileDir"></param>
        /// <returns></returns>
        public async Task<List<Models.Yolov7Predict>> InferenceAsync(string fileDir)
        {
            using var image = Image.Load<Rgb24>(fileDir);
            return await InferenceAsync(image);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="stream"></param>
        /// <returns></returns>
        public async Task<List<Models.Yolov7Predict>> InferenceAsync(Stream stream)
        {
            using var image = Image.Load<Rgb24>(stream);
            return await InferenceAsync(image);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        public async Task<List<Models.Yolov7Predict>> InferenceAsync(Image<Rgb24> image)
        {
            var tensorFeed = PreProcess.Image2DenseTensor(image);
            return await InferenceAsync(tensorFeed);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public async Task<List<Models.Yolov7Predict>> InferenceAsync(DenseTensor<float> tensor)
        {
            var imageShape = tensor.Dimensions[1..].ToArray();
            var lettered = PreProcess.LetterBox(tensor, false, false, true, Stride, new[] { _inputShape[2], _inputShape[3] });

            lettered.Item1 = Operators.Operators.Div(lettered.Item1, 255f);

            // var revert = Operators.Operators.Mul(lettered.Item1.Clone(), 255f);
            //
            // Image<Rgb24> draw = new Image<Rgb24>(Configuration.Default, 640, 640);
            // draw.ProcessPixelRows(accessor =>
            // {
            //     for (var y = 0; y < revert.Dimensions[1]; y++)
            //     {
            //         var pixelSpan = accessor.GetRowSpan(y);
            //         for (var x = 0; x < revert.Dimensions[2]; x++)
            //         {
            //             pixelSpan[x].R = (byte)revert[0, y, x];
            //             pixelSpan[x].G = (byte)revert[1, y, x];
            //             pixelSpan[x].B = (byte)revert[2, y, x];
            //         }
            //     }
            // });
            // await draw.SaveAsync("D:/Documents/Dotnet/BlazorApp1/debugImage.jpg");

            var letteredItem1Dim = lettered.Item1.Dimensions.ToArray();
            long[] newDim = new[] { 1L, letteredItem1Dim[0], letteredItem1Dim[1], letteredItem1Dim[2] };

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, Operators.Operators.ExpandDim(lettered.Item1).Buffer, newDim);
            var inputs = new Dictionary<string, OrtValue> { { _inputNames.First(), inputOrtValue } };

            using var fromResult = await Task.FromResult(_session.Run(_runOptions, inputs, _outputNames));

            float[] resultArrays = fromResult.First().Value.GetTensorDataAsSpan<float>().ToArray();

            Models.Predictions predictions = new Models.Predictions(resultArrays, _categories.ToArray(), new List<float[]>() { lettered.Item3 }, new List<float[]>() { lettered.Item2 }, new List<int[]>() { imageShape });
            return predictions.GetDetect();
        }

        /// <summary>
        /// GetAvailableProviders
        /// </summary>
        /// <returns></returns>
        public List<string> GetAvailableProviders()
        {
            return OrtEnv.Instance().GetAvailableProviders().ToList();
        }

        public void SetExcutionProvider(ExecutionProvider ex)
        {
            switch (ex)
            {
                case ExecutionProvider.CPU:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    break;
                }
                case ExecutionProvider.CUDA:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    OrtCUDAProviderOptions providerOptions = new OrtCUDAProviderOptions();
                    var providerOptionsDict = new Dictionary<string, string>
                    {
                        ["cudnn_conv_use_max_workspace"] = "1",
                        ["device_id"] = "0"
                    };
                    providerOptions.UpdateOptions(providerOptionsDict);
                    _sessionOptions.AppendExecutionProvider_CUDA(providerOptions);
                    break;
                }
                case ExecutionProvider.TensorRT:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    OrtTensorRTProviderOptions provider = new OrtTensorRTProviderOptions();
                    var providerOptionsDict = new Dictionary<string, string>
                    {
                        ["cudnn_conv_use_max_workspace"] = "1",
                        ["device_id"] = "0",
                        ["ORT_TENSORRT_FP16_ENABLE"] = "true",
                        ["ORT_TENSORRT_LAYER_NORM_FP32_FALLBACK"] = "true",
                        ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "true",
                    };
                    provider.UpdateOptions(providerOptionsDict);
                    _sessionOptions.AppendExecutionProvider_Tensorrt(provider);
                    break;
                }
                case ExecutionProvider.Azure:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    break;
                }
                case ExecutionProvider.CoreML:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    break;
                }
                case ExecutionProvider.DML:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    _sessionOptions.EnableMemoryPattern = false;
                    _sessionOptions.AppendExecutionProvider_DML(0);
                    break;
                }
                case ExecutionProvider.NNAPI:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    break;
                }
                case ExecutionProvider.OpenCL:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    break;
                }
                case ExecutionProvider.QNN:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    break;
                }
                case ExecutionProvider.XNNPACK:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    break;
                }
                case ExecutionProvider.OpenVINO:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    _sessionOptions.AppendExecutionProvider_OpenVINO();
                    _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
                    break;
                }
                case ExecutionProvider.oneDNN:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    break;
                }
                case ExecutionProvider.DNNL:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    _sessionOptions.AppendExecutionProvider_Dnnl();
                    break;
                }
                case ExecutionProvider.ROCm:
                {
                    _sessionOptions = CopyOptions(_sessionOptions);
                    OrtROCMProviderOptions provider = new();
                    var providerOptionsDict = new Dictionary<string, string>
                    {
                        ["device_id"] = "0",
                        ["cudnn_conv_use_max_workspace"] = "1"
                    };
                    provider.UpdateOptions(providerOptionsDict);
                    _sessionOptions.AppendExecutionProvider_ROCm(provider);
                    break;
                }
            }

            return;

            SessionOptions CopyOptions(SessionOptions sessionOption)
            {
                SessionOptions sessionOptions = new SessionOptions();
                sessionOptions.EnableCpuMemArena = sessionOption.EnableCpuMemArena;
                sessionOptions.EnableMemoryPattern = sessionOption.EnableMemoryPattern;
                sessionOptions.EnableProfiling = sessionOption.EnableProfiling;
                sessionOptions.ExecutionMode = sessionOption.ExecutionMode;
                sessionOptions.GraphOptimizationLevel = sessionOption.GraphOptimizationLevel;

                return sessionOptions;
            }
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
                TheLogger("[ImageClassifyService][Dispose] disposed");
            }

            _disposed = true;
        }

        /// <summary>
        /// support for bold web and other
        /// </summary>
        /// <param name="message"></param>
        private void TheLogger(string message)
        {
            if (_jsRuntime is not null)
            {
                _jsRuntime.InvokeVoidAsync("console.log", message);
            }

            if (_logger is not null)
            {
                _logger.LogInformation(message);
            }
        }
    }
}