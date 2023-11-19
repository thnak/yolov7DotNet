using Numpy;

namespace yolov7DotNet;

public class KalmanFilter
{
    // private static readonly float[] Chi2Inv95 = new[] { 3.8415f, 5.9915f, 7.8147f, 9.4877f, 11.070f, 12.592f, 14.067f, 15.507f, 16.919f };
    private const int Ndim = 4;
    private const int Dt = 1;
    private static readonly NDarray MotionMat = np.eye(2 * Ndim, 2 * Ndim);
    private static readonly NDarray UpdateMat = np.eye(Ndim, 2 * Ndim);
    private static readonly float StdWeightPosition = 1f / 20;
    private static readonly float StdWeightVelocity = 1f / 160;

    public KalmanFilter()
    {
        for (var x = 0; x < Ndim; x++)
        {
            MotionMat[x, Ndim + x] = np.array(Dt);
        }
    }

    public static (NDarray, NDarray) Initiate(NDarray measurement)
    {
        NDarray meanPos = measurement.copy();
        NDarray meanVel = np.zeros_like(meanPos);
        NDarray mean = np.concatenate((meanPos, meanVel)).flatten();
        var std = new NDarray(new[]
        {
            2 * StdWeightPosition * measurement[0],
            2 * StdWeightPosition * measurement[1], 1 * measurement[2],
            2 * StdWeightPosition * measurement[3],
            10 * StdWeightVelocity * measurement[0],
            10 * StdWeightVelocity * measurement[1],
            0.1f * measurement[2], 10 * StdWeightVelocity * measurement[3]
        });

        NDarray covariance = np.diag(std.square());
        return (mean, covariance);
    }

    public (NDarray, NDarray) Predict(NDarray mean, NDarray covariance)
    {
        NDarray stdPos = new NDarray(new[]
        {
            StdWeightPosition * mean[0].item<float>(),
            StdWeightPosition * mean[1].item<float>(),
            1 * mean[2].item<float>(),
            StdWeightPosition * mean[3].item<float>()
        });
        NDarray stdVel = new NDarray(new[]
        {
            StdWeightVelocity * mean[0].item<float>(),
            StdWeightVelocity * mean[1].item<float>(),
            0.1f * mean[2].item<float>(),
            StdWeightVelocity * mean[3].item<float>()
        });
        NDarray motionCov = np.diag(np.concatenate((stdPos, stdVel)).flatten().square());
        NDarray doted = MotionMat.dot(MotionMat.T);
        covariance = np.linalg.multi_dot(doted, covariance) + motionCov;
        return (mean, covariance);
    }

    public (NDarray, NDarray) Project(NDarray mean, NDarray covariance, float confidence = 0f)
    {
        NDarray std = new NDarray(new[]
        {
            (1 - confidence) * StdWeightPosition * mean[3].item<float>(),
            (1 - confidence) * StdWeightPosition * mean[3].item<float>(),
            (1 - confidence) * 1e-1f,
            (1 - confidence) * StdWeightPosition * mean[3].item<float>()
        });

        NDarray innovationCov = new NDarray(np.diag(std.square()));
        NDarray mean2 = UpdateMat.dot(mean);

        NDarray updatemat = new NDarray(np.eye(4, 4));

        updatemat[0, 0] = covariance[0, 0];
        updatemat[1, 1] = covariance[1, 1];
        updatemat[2, 2] = covariance[2, 2];
        updatemat[3, 3] = covariance[3, 3];

        covariance = updatemat;

        return (mean2, covariance + innovationCov);
    }

    public (NDarray, NDarray) Update(NDarray mean, NDarray covariance, NDarray measurement, float confidence = 0f)
    {
        var project = Project(mean, covariance, confidence);

        NDarray projectedMean = project.Item1;
        NDarray projectedCov = project.Item2;

        NDarray cholFactor = ChoFactor(projectedCov);
        NDarray kalmanGain = ChoSolve(cholFactor, covariance.dot(UpdateMat.T));

        NDarray innovation = measurement - projectedMean;
        NDarray newMean = mean + innovation.dot(kalmanGain.T);
        NDarray newCovariance = covariance - np.linalg.multi_dot(kalmanGain, projectedCov, kalmanGain.T);

        return (newMean, newCovariance);
    }

    public NDarray GatingDistance(NDarray mean, NDarray covariance, NDarray measurements, bool onlyPosition = false)
    {
        var pro = Project(mean, covariance);
        mean = pro.Item1;
        covariance = pro.Item2;
        if (onlyPosition)
        {
            mean = mean[":2"];
            covariance = covariance[":2, :2"];
            measurements = measurements[":, :2"];
        }

        NDarray choleskyFactor = np.linalg.cholesky(covariance);
        NDarray d = measurements - mean;
        NDarray z = SolveTriangular(choleskyFactor, d.T);
        NDarray squaredMaha = (z * z).sum(axis: 0);
        return squaredMaha;
    }

    public NDarray SolveTriangular(NDarray a, NDarray b, bool lower = true)
    {
        int n = a.shape[0];
        float[] x = new float[n];
        if (lower)
        {
            for (var i = 0; i < n; i++)
            {
                x[i] = b[i].item<float>();
                for (var j = 0; j < i; j++)
                {
                    x[i] -= a[i, j].item<float>() * x[j];
                }

                x[i] /= a[i, i].item<float>();
            }
        }
        else
        {
            for (var i = n - 1; i >= 0; i--)
            {
                x[i] = b[i].item<float>();
                for (var j = i + 1; j < n; j++)
                {
                    x[i] -= a[j, i].item<float>() * x[j];
                }

                x[i] /= a[i, i].item<float>();
            }
        }

        return x;
    }

    private NDarray ChoFactor(NDarray array)
    {
        float[,] array2 =
        {
            { 0, 0, 0, 0 },
            { 0, 0, 0, 0 },
            { 0, 0, 0, 0 },
            { 0, 0, 0, 0 },
        };
        for (var i = 0; i < 4; i++)
        {
            for (var j = 0; j < i + 1; j++)
            {
                float sum = 0f;
                for (int k = 0; k < j + 1; k++)
                {
                    sum += array2[i, k] * array2[j, k];
                }

                if (i == j)
                {
                    array2[i, j] = (float)Math.Sqrt(array[i, j].item<float>() - sum);
                }
                else
                {
                    array2[i, j] = (1.0f / array2[j, j]) * (array[i, j].item<float>() - sum);
                }
            }
        }

        NDarray choFactor = new NDarray(np.array(array2));
        return choFactor;
    }

    public NDarray ChoSolve(NDarray l, NDarray b)
    {
        int n = 4;
        float[,] result = new float[8, n];
        for (var xx = 0; xx < 8; xx++)
        {
            float[] y = new float[n];
            for (int i = 0; i < n; i++)
            {
                y[i] = b[xx, i].item<float>();
                for (int j = 0; j < i; j++)
                {
                    y[i] -= l[i][j].item<float>() * y[j];
                }

                y[i] /= l[i][i].item<float>();
            }

            for (int i = n - 1; i >= 0; i--)
            {
                result[xx, i] = y[i];
                for (int j = i + 1; j < n; j++)
                {
                    result[xx, i] -= l[j][i].item<float>() * result[xx, j];
                }

                result[xx, i] /= l[i][i].item<float>();
            }
        }

        return result;
    }
}