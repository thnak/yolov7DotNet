using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;
using Microsoft.JSInterop;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using yolov7DotNet.Helper;
using yolov7DotNet.ModelsHelper;
using yolov7DotNet.Operators;

namespace yolov7DotNet;

/// <summary>
/// 
/// </summary>
public abstract class Yolov7NetService
{
    /// <summary>
    /// model weight include in this project
    /// Default is tiny model
    /// </summary>
    public enum Yolov7Weights
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

    public enum StrongSortWeights
    {
        osnet_x0_25_msmt17
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

    public interface IStrongSort
    {
        Task<DenseTensor<float>> InferenceAsync(DenseTensor<float> tensor);
        void SetExcutionProvider(ExecutionProvider executionProvider, StrongSortWeights? strongSortWeights, byte[]? modelBytes);
        List<string> GetAvailableProviders();
    }

    public class StrongSort : IStrongSort, IDisposable
    {
        private readonly string _prefix = Properties.Resources.prefix;
        private SessionOptions _sessionOptions;
        private InferenceSession _session;
        private RunOptions _runOptions;
        private List<string> _categories;
        private IEnumerable<string> _inputNames;
        private IReadOnlyList<string> _outputNames;
        private bool _disposed;
        private int[] _inputShape = new[] { 128, 256 };

        public StrongSort(StrongSortWeights? strongSortWeights, byte[] modelBytes)
        {
            var availableProvider = OrtEnv.Instance().GetAvailableProviders()[0];
            switch (availableProvider)
            {
                case "CUDAExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.CUDA, strongSortWeights, modelBytes);
                    break;
                }
                case "TensorrtExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.TensorRT, strongSortWeights, modelBytes);
                    break;
                }
                case "DNNLExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.DNNL, strongSortWeights, modelBytes);
                    break;
                }
                case "OpenVINOExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.OpenVINO, strongSortWeights, modelBytes);

                    break;
                }
                case "DmlExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.DML, strongSortWeights, modelBytes);
                    break;
                }
                case "ROCMExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.ROCm, strongSortWeights, modelBytes);
                    break;
                }
                default:
                {
                    SetExcutionProvider(ExecutionProvider.CPU, strongSortWeights, modelBytes);
                    break;
                }
            }
        }

        public Task<DenseTensor<float>> InferenceAsync(DenseTensor<float> tensor)
        {
            throw new NotImplementedException();
        }

        public void SetExcutionProvider(ExecutionProvider executionProvider, StrongSortWeights? strongSortWeights, byte[]? modelBytes)
        {
        }

        /// <summary>
        /// GetAvailableProviders
        /// </summary>
        /// <returns></returns>
        public List<string> GetAvailableProviders()
        {
            return OrtEnv.Instance().GetAvailableProviders().ToList();
        }

        /// <summary>
        /// release unused resource
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
                _sessionOptions.Dispose();
                _session.Dispose();
                _runOptions.Dispose();
            }

            _disposed = true;
        }
    }

    public interface IYolov7
    {
        Task<List<Models.Yolov7Predict>> InferenceAsync(MemoryStream memoryStream);
        Task<List<Models.Yolov7Predict>> InferenceAsync(Image<Rgb24> image);
        Task<List<Models.Yolov7Predict>> InferenceAsync(string fileDir);
        Task<List<Models.Yolov7Predict>> InferenceAsync(Stream stream);
        Task<List<Models.Yolov7Predict>> InferenceAsync(DenseTensor<float> tensor);
        List<string> GetAvailableProviders();
        void SetExcutionProvider(ExecutionProvider ex, Yolov7Weights? weight, byte[]? byteWeight);
        Task<float> WarmUp(int cycle,int batchSize);
    }

    /// <summary>
    /// 
    /// </summary>
    public class Yolov7 : IYolov7, IDisposable
    {
        private readonly string _prefix = Properties.Resources.prefix;
        private SessionOptions _sessionOptions;
        private InferenceSession _session;
        private RunOptions _runOptions;
        private List<string> _categories;
        private IEnumerable<string> _inputNames;
        private IReadOnlyList<string> _outputNames;
        private int Stride { get; set; }
        private bool _disposed;
        private IMemoryCache MemoryCache { get; set; }

        private int[] _inputShape;
        private readonly IJSRuntime? _jsRuntime;
        private readonly ILogger<Yolov7>? _logger;

        public Yolov7() : this(weight: Yolov7NetService.Yolov7Weights.Yolov7Tiny, jsRuntime: null, byteWeight: null, logger: null)
        {
        }

        public Yolov7(Yolov7NetService.Yolov7Weights yolov7Weights) : this(weight: yolov7Weights, jsRuntime: null, byteWeight: null, logger: null)
        {
        }

        public Yolov7(IJSRuntime jsRuntime) : this(weight: Yolov7NetService.Yolov7Weights.Yolov7Tiny, jsRuntime: jsRuntime, byteWeight: null, logger: null)
        {
        }

        public Yolov7(Yolov7NetService.Yolov7Weights yolov7Weights, IJSRuntime jsRuntime) : this(weight: yolov7Weights, jsRuntime: jsRuntime, byteWeight: null, logger: null)
        {
        }

        public Yolov7(Yolov7NetService.Yolov7Weights yolov7Weights, ILogger<Yolov7> logger) : this(weight: yolov7Weights, jsRuntime: null, byteWeight: null, logger: logger)
        {
        }

        public Yolov7(byte[] modelWeights) : this(byteWeight: modelWeights, jsRuntime: null, logger: null, weight: null)
        {
        }

        public Yolov7(byte[] modelWeights, IJSRuntime jsRuntime) : this(byteWeight: modelWeights, jsRuntime: jsRuntime, logger: null)
        {
        }

        public Yolov7(byte[] modelWeights, ILogger<Yolov7> logger) : this(byteWeight: modelWeights, jsRuntime: null, logger: logger, weight: null)
        {
        }

        public Yolov7(IJSRuntime? jsRuntime = null, Yolov7NetService.Yolov7Weights? weight = null, byte[]? byteWeight = null) : this(jsRuntime: jsRuntime, logger: null, weight: weight, byteWeight: byteWeight)
        {
        }


        public Yolov7(IJSRuntime? jsRuntime = null, ILogger<Yolov7>? logger = null, Yolov7NetService.Yolov7Weights? weight = null, byte[]? byteWeight = null)
        {
            if (jsRuntime != null)
            {
                _jsRuntime = jsRuntime;
            }

            if (logger != null)
            {
                _logger = logger;
            }

            var availableProvider = OrtEnv.Instance().GetAvailableProviders()[0];

            switch (availableProvider)
            {
                case "CUDAExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.CUDA, weight, byteWeight);
                    break;
                }
                case "TensorrtExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.TensorRT, weight, byteWeight);
                    break;
                }
                case "DNNLExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.DNNL, weight, byteWeight);
                    break;
                }
                case "OpenVINOExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.OpenVINO, weight, byteWeight);

                    break;
                }
                case "DmlExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.DML, weight, byteWeight);
                    break;
                }
                case "ROCMExecutionProvider":
                {
                    SetExcutionProvider(ExecutionProvider.ROCm, weight, byteWeight);
                    break;
                }
                default:
                {
                    SetExcutionProvider(ExecutionProvider.CPU, weight, byteWeight);
                    break;
                }
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

            var feedTensor = Ops.ExpandDim(Ops.Div(lettered.Item1, 255f));

            return await RunNet(feedTensor, new List<float[]>() { lettered.Item3 }, new List<float[]>() { lettered.Item2 }, new List<int[]>() { imageShape });
        }

        /// <summary>
        /// start inference and return the predictions, support dynamic batch
        /// </summary>
        /// <param name="tensor">4 dimension only</param>
        /// <param name="dhdws"></param>
        /// <param name="ratios"></param>
        /// <param name="imageShapes"></param>
        /// <returns></returns>
        public async Task<List<Models.Yolov7Predict>> RunNet(DenseTensor<float> tensor, List<float[]> dhdws, List<float[]> ratios, List<int[]> imageShapes)
        {
            long[] newDim = new[] { (long)tensor.Dimensions[0], tensor.Dimensions[1], tensor.Dimensions[2], tensor.Dimensions[3] };
            var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, newDim);
            var inputs = new Dictionary<string, OrtValue> { { _inputNames.First(), inputOrtValue } };

            var fromResult = await Task.FromResult(_session.Run(_runOptions, inputs, _outputNames));

            float[] resultArrays = fromResult[0].Value.GetTensorDataAsSpan<float>().ToArray();
            inputOrtValue.Dispose();
            fromResult.Dispose();
            Models.Predictions predictions = new Models.Predictions(resultArrays, _categories.ToArray(), dhdws, ratios, imageShapes);
            return predictions.GetDetect();
        }

        /// <summary>
        /// start inference and return the predictions, support dynamic batch
        /// </summary>
        /// <param name="tensor">4 dimension only</param>
        /// <param name="dhdws"></param>
        /// <param name="ratios"></param>
        /// <param name="imageShapes"></param>
        /// <returns></returns>
        public async Task<List<Models.Yolov7Predict>> RunNet(DenseTensor<Float16> tensor, List<float[]> dhdws, List<float[]> ratios, List<int[]> imageShapes)
        {
            long[] newDim = new[] { (long)tensor.Dimensions[0], tensor.Dimensions[1], tensor.Dimensions[2], tensor.Dimensions[3] };
            var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, newDim);
            var inputs = new Dictionary<string, OrtValue> { { _inputNames.First(), inputOrtValue } };

            var fromResult = await Task.FromResult(_session.Run(_runOptions, inputs, _outputNames));

            float[] resultArrays = fromResult[0].Value.GetTensorDataAsSpan<float>().ToArray();
            inputOrtValue.Dispose();
            fromResult.Dispose();
            Models.Predictions predictions = new Models.Predictions(resultArrays, _categories.ToArray(), dhdws, ratios, imageShapes);
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

        /// <summary>
        /// set 
        /// </summary>
        /// <param name="ex"></param>
        /// <param name="weight"></param>
        /// <param name="byteWeight"></param>
        /// <exception cref="Exception"></exception>
        public void SetExcutionProvider(Yolov7NetService.ExecutionProvider ex, Yolov7NetService.Yolov7Weights? weight, byte[]? byteWeight)
        {
            switch (ex)
            {
                case ExecutionProvider.CPU:
                {
                    _sessionOptions = DefaultOptions();
                    TheLogger($"[{_prefix}][INIT][ExecutionProvider][CPU]");
                    break;
                }
                case ExecutionProvider.CUDA:
                {
                    _sessionOptions = DefaultOptions();
                    OrtCUDAProviderOptions providerOptions = new OrtCUDAProviderOptions();
                    var providerOptionsDict = new Dictionary<string, string>
                    {
                        ["cudnn_conv_use_max_workspace"] = "1",
                        ["device_id"] = "0"
                    };
                    providerOptions.UpdateOptions(providerOptionsDict);
                    _sessionOptions.AppendExecutionProvider_CUDA(providerOptions);
                    TheLogger($"[{_prefix}][INIT][ExecutionProvider][CUDA]");
                    break;
                }
                case ExecutionProvider.TensorRT:
                {
                    _sessionOptions = DefaultOptions();
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
                    TheLogger($"[{_prefix}][INIT][ExecutionProvider][Tensorrt]");

                    break;
                }
                case ExecutionProvider.Azure:
                {
                    _sessionOptions = DefaultOptions();
                    break;
                }
                case ExecutionProvider.CoreML:
                {
                    _sessionOptions = DefaultOptions();
                    break;
                }
                case ExecutionProvider.DML:
                {
                    _sessionOptions = DefaultOptions();
                    _sessionOptions.EnableMemoryPattern = false;
                    _sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                    _sessionOptions.AppendExecutionProvider_DML(1);
                    TheLogger($"[{_prefix}][INIT][ExecutionProvider][Dml]");
                    break;
                }
                case ExecutionProvider.NNAPI:
                {
                    _sessionOptions = DefaultOptions();
                    break;
                }
                case ExecutionProvider.OpenCL:
                {
                    _sessionOptions = DefaultOptions();
                    break;
                }
                case ExecutionProvider.QNN:
                {
                    _sessionOptions = DefaultOptions();
                    break;
                }
                case ExecutionProvider.XNNPACK:
                {
                    _sessionOptions = DefaultOptions();
                    break;
                }
                case ExecutionProvider.OpenVINO:
                {
                    _sessionOptions = DefaultOptions();
                    _sessionOptions.AppendExecutionProvider_OpenVINO();
                    _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
                    TheLogger($"[{_prefix}][INIT][ExecutionProvider][OpenVINO]");

                    break;
                }
                case Yolov7NetService.ExecutionProvider.oneDNN:
                {
                    _sessionOptions = DefaultOptions();
                    break;
                }
                case ExecutionProvider.DNNL:
                {
                    _sessionOptions = DefaultOptions();
                    _sessionOptions.AppendExecutionProvider_Dnnl();
                    TheLogger($"[{_prefix}][INIT][ExecutionProvider][DNNL]");

                    break;
                }
                case ExecutionProvider.ROCm:
                {
                    _sessionOptions = DefaultOptions();
                    OrtROCMProviderOptions provider = new();
                    var providerOptionsDict = new Dictionary<string, string>
                    {
                        ["device_id"] = "0",
                        ["cudnn_conv_use_max_workspace"] = "1"
                    };
                    provider.UpdateOptions(providerOptionsDict);
                    _sessionOptions.AppendExecutionProvider_ROCm(provider);
                    TheLogger($"[{_prefix}][INIT][ExecutionProvider][ROCM]");
                    break;
                }
            }

            var prepackedWeightsContainer = new PrePackedWeightsContainer();
            _runOptions = new RunOptions();
            if (weight is not null)
            {
                switch (weight)
                {
                    case Yolov7NetService.Yolov7Weights.Yolov7:
                    {
                        _session = new InferenceSession(Properties.Resources.yolov7, _sessionOptions, prepackedWeightsContainer);
                        TheLogger($"[{_prefix}][INIT][Yolov7Weights][yolov7]");
                        break;
                    }
                    case Yolov7NetService.Yolov7Weights.Yolov7Tiny:
                    {
                        _session = new InferenceSession(Properties.Resources.yolov7_tiny, _sessionOptions, prepackedWeightsContainer);
                        TheLogger($"[{_prefix}][INIT][Yolov7Weights][yolov7_tiny]");
                        break;
                    }
                    default:
                    {
                        _session = new InferenceSession(Properties.Resources.yolov7_tiny, _sessionOptions, prepackedWeightsContainer);
                        TheLogger($"[{_prefix}][INIT][Yolov7Weights][yolov7_tiny]");
                        break;
                    }
                }
            }
            else if (byteWeight is not null)
            {
                _session = new InferenceSession(byteWeight, _sessionOptions, prepackedWeightsContainer);
                TheLogger($"[{_prefix}][INIT][Yolov7Weights][byte model]");
            }

            else
            {
                _session = new InferenceSession(Properties.Resources.yolov7_tiny, _sessionOptions, prepackedWeightsContainer);
                TheLogger($"[{_prefix}][INIT][Yolov7Weights][yolov7_tiny]");
            }

            OrtIoBinding ioBinding = _session.CreateIoBinding();
            _session.RunWithBoundResults(_runOptions, ioBinding);
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
                var exception = new Exception(message: $"[{_prefix}][Init][ERROR] could not init");
                TheLogger($"[{_prefix}][Init][ERROR] could not init");
                exception.HelpLink = "https://github.com/thnak/yolov7DotNet";
                exception.HResult = 0;
                exception.Source = "https://github.com/thnak/yolov7DotNet";
                throw exception;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="cycle">total number loop</param>
        /// <param name="batchSize">tensor batch size</param>
        /// <returns>ElapsedMilliseconds during the entire loop</returns>
        public async Task<float> WarmUp(int cycle, int batchSize)
        {
            Stopwatch sw = Stopwatch.StartNew();
            sw.Start();
            var shape = new[] { batchSize, _inputShape[1], _inputShape[2], _inputShape[3] };


            var dtype = _session.InputMetadata.Values.First().ElementDataType;
            if (dtype == TensorElementType.Float16)
            {
                DenseTensor<Float16> tensor = new DenseTensor<Float16>(shape);
                List<float[]> dwdh = new List<float[]>() { { new[] { 0f, 0 } } };
                List<float[]> ratio = new List<float[]>() { { new[] { 1f, 1 } } };
                List<int[]> imageShape = new List<int[]>() { { new[] { _inputShape[0], _inputShape[1] } } };
                for (int i = 0; i < cycle; i++)
                {
                    await RunNet(tensor, dwdh, ratio, imageShape);
                }

                sw.Stop();
                return sw.ElapsedMilliseconds;
            }
            else
            {
                DenseTensor<float> tensor = new DenseTensor<float>(shape);
                List<float[]> dwdh = new List<float[]>() { { new[] { 0f, 0 } } };
                List<float[]> ratio = new List<float[]>() { { new[] { 1f, 1 } } };
                List<int[]> imageShape = new List<int[]>() { { new[] { _inputShape[0], _inputShape[1] } } };
                for (int i = 0; i < cycle; i++)
                {
                    await RunNet(tensor, dwdh, ratio, imageShape);
                }

                sw.Stop();
                return sw.ElapsedMilliseconds;
            }
        }

        /// <summary>
        /// release unused resource
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
                _logger.LogInformation("Log {log}", message);
            }
        }
    }

    /// <summary>
    /// init new default SessionOption
    /// </summary>
    /// <returns></returns>
    private static SessionOptions DefaultOptions()
    {
        var sessionOptions = new SessionOptions();
        sessionOptions.EnableMemoryPattern = true;
        sessionOptions.EnableCpuMemArena = true;
        sessionOptions.EnableProfiling = false;
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
        sessionOptions.OptimizedModelFilePath = "D:\\Documents\\GitHub\\yolov7DotNet\\yolov7DotNet.onnx";
        return sessionOptions;
    }
}