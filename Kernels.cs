using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Algorithms;

namespace CarDeepQ
{
    public static class Kernels
    {
        public static Context Context;
        public static Accelerator Accelerator => Kernels2.Accelerator;

        public static readonly Action<Index1D, ArrayView<float>, float, ArrayView<float>> VectorMultConstant;
        public static readonly Action<Index1D, ArrayView<float>, float, ArrayView<float>> VectorDivConstant;
        public static readonly Action<Index1D, ArrayView<float>, float, ArrayView<float>> VectorInverseConst;
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> VectorInverse;
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> VectorSqrt;
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> VectorInverseSqrt;
                      
                      
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> VectorMult;
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> VectorAdd;
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> VectorSub;
                      
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> VectorSigmoid;
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> VectorSigmoidPrime;
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> VectorReLU;
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> VectorReLUPrime;
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> VectoreLU;
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> VectoreLUPrime;
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> VectorLinear;
        public static readonly Action<Index1D, ArrayView<float>, ArrayView<float>> VectorLinearPrime;
                      
        public static readonly Action<Index1D, ArrayView<float>, float> SetValue;
        public static readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, float> SetValue2D;
                      
        public static readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>> MatrixVectorMult;
        public static readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>> MatrixVectorMult2;
        public static readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>, ArrayView2D<float, Stride2D.DenseX>> MatrixSetVectorSqrdMult;
                      
        public static readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, float, ArrayView2D<float, Stride2D.DenseX>> MatrixDivConst;
        public static readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, ArrayView1D<float, Stride1D.Dense>> SetMovingAverageBiases;
        public static readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, ArrayView1D<float, Stride1D.Dense>> SetBiasesTrain;
                      
        public static readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float, ArrayView2D<float, Stride2D.DenseX>> SetMovingAverage;
        public static readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float, ArrayView2D<float, Stride2D.DenseX>> SetWeightsTrain;

        public static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> SetLastLayerError;


        static Kernels()
        {
            /*Context.CreateDefault();
            Context = Context.Create(
                (builder) => { builder.Default(); builder.EnableAlgorithms(); });*/
            
            //Accelerator = Context.CreateCudaAccelerator(0);

            
            VectorMultConstant = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float, ArrayView<float>>(VectorMultConstKernel);
            VectorDivConstant = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float, ArrayView<float>>(VectorDivConstKernel);
            VectorInverseConst = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float, ArrayView<float>>(VectorInverseConstKernel);
            VectorInverse = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(VectorInverseKernel);
            VectorSqrt = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(VectorSqrtKernel);
            VectorInverseSqrt = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(VectorInverseSqrtKernel);
            VectorMult = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorMultKernel);
            VectorAdd = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);
            VectorSub = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorSubKernel);

            VectorSigmoid = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(VectorApplyFunctionKernel<Sigmoid>);
            VectorSigmoidPrime = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(VectorApplyFunctionKernel<SigmoidPrime>);
            VectorReLU = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(VectorApplyFunctionKernel<ReLU>);
            VectorReLUPrime = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(VectorApplyFunctionKernel<ReLUPrime>);
            VectoreLU = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(VectoreLUKernel);
            VectoreLUPrime = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(VectoreLUPrimeKernel);
            VectorLinear = (a, b, c) => { };
            VectorLinearPrime = (a, b, c) => SetValue(a, b, 1);

            SetValue = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float>(SetValueKernel);
            SetValue2D = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, float>(SetValue2DKernel);

            MatrixVectorMult = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>>(MatrixVectorMultKernel);
            MatrixVectorMult2 = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>>(MatrixVectorMultKernel2);
            MatrixSetVectorSqrdMult = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>, ArrayView2D<float, Stride2D.DenseX>>(MatrixSetVectorSqrdMultKernel);

            MatrixVectorMult = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>>(MatrixVectorMultKernel);
            MatrixSetVectorSqrdMult = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>, ArrayView2D<float, Stride2D.DenseX>>(MatrixSetVectorSqrdMultKernel);
            MatrixDivConst = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, float, ArrayView2D<float, Stride2D.DenseX>>(MatrixDivConstKernel);
            SetMovingAverageBiases = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, ArrayView1D<float, Stride1D.Dense>>(SetMovingAverageBiasesKernel);
            SetBiasesTrain = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, ArrayView1D<float, Stride1D.Dense>>(SetBiasesTrainKernel);
            SetMovingAverage = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float, ArrayView2D<float, Stride2D.DenseX>>(SetMovingAverageKernel);
            SetWeightsTrain = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float, ArrayView2D<float, Stride2D.DenseX>>(SetWeightsTrainKernel);
            
            SetLastLayerError = Accelerator.LoadAutoGroupedStreamKernel <Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(SetLastLayerErrorKernel);


            Accelerator.Synchronize();
        }

        private static readonly Random rand = new Random();

        private static void VectorMultConstKernel(Index1D i, ArrayView<float> data, float constant, ArrayView<float> output)
            => output[i] = data[i] * constant;

        private static void VectorDivConstKernel(Index1D i, ArrayView<float> data, float constant, ArrayView<float> output)
            => output[i] = data[i] / constant;

        private static void VectorInverseConstKernel(Index1D i, ArrayView<float> data, float constant, ArrayView<float> output)
            => output[i] = constant / data[i];

        private static void VectorInverseKernel(Index1D i, ArrayView<float> data, ArrayView<float> output)
            => output[i] = 1 / data[i];

        private static void VectorSqrtKernel(Index1D i, ArrayView<float> data, ArrayView<float> output)
            => output[i] = XMath.Sqrt(data[i]);

        private static void VectorInverseSqrtKernel(Index1D i, ArrayView<float> data, ArrayView<float> output)
            => output[i] = XMath.Rsqrt(data[i]);

        private static void VectorMultKernel(Index1D i, ArrayView<float> data, ArrayView<float> data2, ArrayView<float> output)
            => output[i] = data[i] * data2[i];

        private static void VectorAddKernel(Index1D i, ArrayView<float> data, ArrayView<float> data2, ArrayView<float> output)
            => output[i] = data[i] + data2[i];

        private static void VectorSubKernel(Index1D i, ArrayView<float> data, ArrayView<float> data2, ArrayView<float> output)
            => output[i] = data[i] - data2[i];

        private static void SetValueKernel(Index1D i, ArrayView<float> data, float value)
            => data[i] = value;

        private static void SetValue2DKernel(Index2D i, ArrayView2D<float, Stride2D.DenseX> data, float value)
            => data[i.X, i.Y] = value;

        private static void MatrixVectorMultKernel(Index2D i, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView<float> vector, ArrayView<float> output)
            => Atomic.Add(ref output[i.X], matrix[i.X, i.Y] * vector[i.Y]);
        //We use Atomic because it solves the problem of multiple threads writing to the same variable

        private static void MatrixVectorMultKernel2(Index2D i, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView<float> vector, ArrayView<float> output)
            => Atomic.Add(ref output[i.Y], matrix[i.X, i.Y] * vector[i.X]);


        private static void VectorApplyFunctionKernel<TFunc>(Index1D i, ArrayView<float> data, ArrayView<float> output) where TFunc : struct, IFloatFunc
        {
            var function = default(TFunc);
            output[i] = function.Apply(data[i]);
        }

        private static void VectoreLUKernel(Index1D i, ArrayView<float> data, ArrayView<float> output)
        {
            if (output[i] <= 0)
                output[i] = (float)(0.1f * (XMath.Exp(output[i]) - 1));
        }

        private static void VectoreLUPrimeKernel(Index1D i, ArrayView<float> data, ArrayView<float> output)
        {
            if (output[i] > 0)
                output[i] = 1;
            else
                output[i] = (float)(0.1f * (XMath.Exp(output[i]) - 1)) + 0.1f;
        }

        private static void MatrixSetVectorSqrdMultKernel(Index2D i, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView<float> vectorX, ArrayView<float> vectorY, ArrayView2D<float, Stride2D.DenseX> output)
            => output[i.X, i.Y] = matrix[i.X, i.Y] + vectorX[i.X] * vectorY[i.Y];


        private static void SetMovingAverageBiasesKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> movingAverage, ArrayView1D<float, Stride1D.Dense> move, float beta, ArrayView1D<float, Stride1D.Dense> output)
            => output[i] = beta * movingAverage[i] + (1 - beta) * move[i] * move[i];
        //MovingAverageBiases[l][n] = Beta * MovingAverageBiases[l][n] + (1 - Beta) * moveBiases[l][n] * moveBiases[l][n];
        

        public static void SetBiasesTrainKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> biases, ArrayView1D<float, Stride1D.Dense> move, ArrayView1D<float, Stride1D.Dense> movingAverage, float learningRate, ArrayView1D<float, Stride1D.Dense> output)
            => output[i] = biases[i] - move[i] * learningRate * XMath.Rsqrt(movingAverage[i]);
        //Biases[l][n] -= moveBiases[l][n] * (LearningRate / (float)Math.Sqrt(MovingAverageBiases[l][n]));


        private static void MatrixDivConstKernel(Index2D i, ArrayView2D<float, Stride2D.DenseX> matrix, float constant, ArrayView2D<float, Stride2D.DenseX> output)
            => output[i] = matrix[i] / constant;

        private static void SetMovingAverageKernel(Index2D i, ArrayView2D<float, Stride2D.DenseX> movingAverage, ArrayView2D<float, Stride2D.DenseX> move, float beta, ArrayView2D<float, Stride2D.DenseX> output)
            => output[i] = beta * movingAverage[i] + (1 - beta) * move[i] * move[i];
        //MovingAverage[l][n][prevN] = Beta * MovingAverage[l][n][prevN] + (1 - Beta) * moveWeights[l][n][prevN] * moveWeights[l][n][prevN];

        public static void SetWeightsTrainKernel(Index2D i, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> move, ArrayView2D<float, Stride2D.DenseX> movingAverage, float learningRate, ArrayView2D<float, Stride2D.DenseX> output)
           => output[i] = weights[i] - move[i] * learningRate * XMath.Rsqrt(movingAverage[i]);
        //Weights[l][n][prevN] -= moveWeights[l][n][prevN] * (LearningRate / (float)Math.Sqrt(MovingAverage[l][n][prevN]));

        public static void SetLastLayerErrorKernel(Index1D i, ArrayView<float> nnOutput, ArrayView<float> target, ArrayView<float> zFunctionned, ArrayView<float> output)
            => output[i.X] = 2 * (nnOutput[i] - target[i]) * zFunctionned[i];


        static readonly float[] layers = new float[5] { 14, 32, 32, 32, 6 };
        public static void WholeTraining()
        {
            



        }




        /*public static void ForwardLayerEkuKernel(Index2D i, ArrayView<float> input, ArrayView<float> biases, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView<float> outputZ, ArrayView<float> outputNeurons)
        {

            Atomic.Add(outputZ[i.X], weights[i.X, i.Y] * input[i.X]);
        }*/

        public static void MatrixMult(ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView<float> vector, ArrayView<float> output)
        {
            for(int i = 0; i < matrix.Extent.X; i++)
            {

            }
        }

        public interface IFloatFunc
        {
            float Apply(float x);
        }

        private struct Sigmoid : Kernels.IFloatFunc
        {
            public float Apply(float x)
                => (float)(1 / (1 + XMath.Exp(-x)));
        }

        private struct SigmoidPrime : Kernels.IFloatFunc
        {
            public float Apply(float x)
                => (float)(1 / (1 + XMath.Exp(-x))) * (1 - (float)(1 / (1 + XMath.Exp(-x))));
        }

        private struct ReLU : Kernels.IFloatFunc
        {
            public float Apply(float x)
            {
                if (x >= 0)
                    return x;
                return 0;
            }
        }

        private struct ReLUPrime : Kernels.IFloatFunc
        {
            public float Apply(float x)
            {
                if (x > 0)
                    return 1;
                return 0;
            }
        }

        private struct eLU : Kernels.IFloatFunc
        {
            public float Apply(float x)
            {
                if (x > 0)
                    return x;
                return (float)(0.1f * (XMath.Exp(x) - 1));
            }
        }

        private struct eLUPrime : Kernels.IFloatFunc
        {
            public float Apply(float x)
            {
                if (x > 0)
                    return 1;
                return (float)(0.1f * (XMath.Exp(x) - 1)) + 0.1f;
            }
        }

        private struct Linear : Kernels.IFloatFunc
        {
            public float Apply(float x)
                => x;
        }

        private struct LinearPrime : Kernels.IFloatFunc
        {
            public float Apply(float x)
                => 1;
        }
    }
}
