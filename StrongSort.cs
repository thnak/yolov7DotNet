using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace yolov7DotNet;
public interface IStrongSort
{
    Task<DenseTensor<float>> InferenceAsync(DenseTensor<float> tensor);
    void SetExcutionProvider(Yolov7NetService.ExecutionProvider executionProvider, Yolov7NetService.StrongSortWeights? strongSortWeights, byte[]? modelBytes);
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
        private readonly int[] _inputShape = new[] { 128, 256 };

        public StrongSort(Yolov7NetService.StrongSortWeights? strongSortWeights, byte[] modelBytes)
        {
            var availableProvider = OrtEnv.Instance().GetAvailableProviders()[0];
            switch (availableProvider)
            {
                case "CUDAExecutionProvider":
                {
                    SetExcutionProvider(Yolov7NetService.ExecutionProvider.CUDA, strongSortWeights, modelBytes);
                    break;
                }
                case "TensorrtExecutionProvider":
                {
                    SetExcutionProvider(Yolov7NetService.ExecutionProvider.TensorRT, strongSortWeights, modelBytes);
                    break;
                }
                case "DNNLExecutionProvider":
                {
                    SetExcutionProvider(Yolov7NetService.ExecutionProvider.DNNL, strongSortWeights, modelBytes);
                    break;
                }
                case "OpenVINOExecutionProvider":
                {
                    SetExcutionProvider(Yolov7NetService.ExecutionProvider.OpenVINO, strongSortWeights, modelBytes);

                    break;
                }
                case "DmlExecutionProvider":
                {
                    SetExcutionProvider(Yolov7NetService.ExecutionProvider.DML, strongSortWeights, modelBytes);
                    break;
                }
                case "ROCMExecutionProvider":
                {
                    SetExcutionProvider(Yolov7NetService.ExecutionProvider.ROCm, strongSortWeights, modelBytes);
                    break;
                }
                default:
                {
                    SetExcutionProvider(Yolov7NetService.ExecutionProvider.CPU, strongSortWeights, modelBytes);
                    break;
                }
            }
        }

        public Task<DenseTensor<float>> InferenceAsync(DenseTensor<float> tensor)
        {
            throw new NotImplementedException();
        }

        public void SetExcutionProvider(Yolov7NetService.ExecutionProvider executionProvider, Yolov7NetService.StrongSortWeights? strongSortWeights, byte[]? modelBytes)
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
