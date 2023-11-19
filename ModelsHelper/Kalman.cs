using Microsoft.ML.OnnxRuntime.Tensors;

namespace yolov7DotNet.ModelsHelper
{
    internal class Kalman
    {
        private static readonly float[] chi2inv95 = new[] { 3.8415f, 5.9915f, 7.8147f, 9.4877f, 11.070f, 12.592f, 14.067f, 15.507f, 16.919f };
        private static int ndim = 4;
        private static int dt = 1;
        private static DenseTensor<float> _motion_mat = Operators.Ops.Eye(new[] { 2 * ndim, 2 * ndim });
        private static DenseTensor<float> _update_mat = Operators.Ops.Eye(new[] { ndim, 2 * ndim });
        private static float _std_weight_position = 1f / 20;
        private static float _std_weight_velocity = 1f / 160;
        private DenseTensor<float> _motion_mat_dual = Operators.Ops.Eye(new[] { 2 * ndim, 2 * ndim });

        public Kalman()
        {
            for (int i = 0; i < ndim; i++)
            {
                _motion_mat[i, ndim + 1] = dt;
                _motion_mat_dual[i, ndim + 1] = dt;
                _motion_mat_dual[ndim + 1, i] = dt;
            }
        }

        public (DenseTensor<float>, DenseTensor<float>) initiate(DenseTensor<float> measurement)
        {
            DenseTensor<float> mean_pos = measurement;
            DenseTensor<float> mean_vel = new DenseTensor<float>(measurement.Dimensions);

            DenseTensor<float> mean_ = Operators.Ops.concatenateFlatten(mean_pos, mean_vel);
            DenseTensor<float> std = new DenseTensor<float>(new ReadOnlySpan<int>(new []{8}));

            std[0] = 2 * _std_weight_position * measurement[0];
            std[1] = 2 * _std_weight_position * measurement[1];
            std[2] = measurement[2];
            std[3] = 2 * _std_weight_position * measurement[3];
            std[4] = 10 * _std_weight_velocity * measurement[0];
            std[5] = 10 * _std_weight_velocity * measurement[1];
            std[6] = 0.1f * measurement[2];
            std[7] = 10 * _std_weight_velocity * measurement[3];
            DenseTensor<float> covariance = Operators.Ops.Diag(Operators.Ops.Square(std));

            return (mean_, covariance);
        }

        public (DenseTensor<float>, DenseTensor<float>) predict(DenseTensor<float> mean, DenseTensor<float> covariance)
        {
            DenseTensor<float> std_pos = new DenseTensor<float>(new ReadOnlySpan<int>(new[] { 4 }));
            DenseTensor<float> std_vel = new DenseTensor<float>(new ReadOnlySpan<int>(new[] { 4 }));

            std_pos[0] = _std_weight_position * mean[0];
            std_pos[1] = _std_weight_position * mean[1];
            std_pos[2] = mean[2];
            std_pos[3] = _std_weight_position * mean[0];

            std_vel[0] = _std_weight_velocity * mean[0];
            std_vel[1] = mean[2];
            std_vel[2] = _std_weight_velocity * mean[1];
            std_vel[3] = _std_weight_velocity * mean[3];

            DenseTensor<float> motion_cov = Operators.Ops.Diag(Operators.Ops.concatenateFlatten(std_pos, std_vel));

            covariance = Operators.Ops.Mul(_motion_mat_dual, covariance);

            return (mean, covariance);
        }

        public static (DenseTensor<float>, DenseTensor<float>) project(DenseTensor<float> mean, DenseTensor<float> covariance, float confidence = 0f)
        {
            DenseTensor<float> std_ = new DenseTensor<float>(new ReadOnlySpan<int>(new []{4}));
            std_.Fill(_std_weight_position * mean[3]);
            std_[2] = 0.1f;

            for (int i = 0; i < std_.Length; i++)
            {
                std_[i] *= 1 - confidence;
            }


            DenseTensor<float> innovation_cov = Operators.Ops.Diag(Operators.Ops.Square(std_));
            mean = Operators.Ops.Mul(_update_mat, mean);

            DenseTensor<float> updatemat = new DenseTensor<float>(new ReadOnlySpan<int>(new[] { 4, 4 }));

            updatemat[0, 0] = covariance[0, 0];
            updatemat[1, 1] = covariance[1, 1];
            updatemat[2, 2] = covariance[2, 2];
            updatemat[3, 3] = covariance[3, 3];

            updatemat = Operators.Ops.Add(updatemat, innovation_cov);

            return (mean, updatemat);
        }

        public static (DenseTensor<float>, DenseTensor<float>) update(DenseTensor<float> mean, DenseTensor<float> covariance, DenseTensor<float> measurement, float confidence = 0f)
        {
            var _project = project(mean, covariance);

            DenseTensor<float> projected_mean = _project.Item1;
            DenseTensor<float> projected_cov = _project.Item2;

            return (projected_cov, projected_cov);
        }
    }
}