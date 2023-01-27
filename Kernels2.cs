using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CarDeepQ
{
    public static class Kernels2
    {
        public static Context Context;
        public static Accelerator Accelerator;

        static Kernels2()
        {
            Context.CreateDefault();
            Context = Context.Create(
                (builder) => { builder.Default(); builder.EnableAlgorithms(); });
            Accelerator = Context.GetPreferredDevice(preferCPU:false).CreateAccelerator(Context);
            

            //Accelerator = Context.CreateCudaAccelerator(0);
        
            ForwardLayer = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>>(ForwardLayerKernel);
            eLU = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(eLUKernel);
            eLUPrime = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(eLUPrimeKernel);


            Linear = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SetKernel);
            LinearPrime = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(Set1Kernel);

            Set = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SetKernel);


            ModifyBiases = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float, float>(ModifyBiasesKernel);
            ModifyWeights = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float, float, float>(ModifyWeightsKernel);

            
            SetLastLayerError = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(SetLastLayerErrorKernel);
            
            MultZFunctionned = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(MultZFunctionnedKernel);

            Accelerator.Synchronize();
        }




        public static Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>> ForwardLayer;
        private static void ForwardLayerKernel(Index2D i, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView<float> neurons, ArrayView<float> output)
            => Atomic.Add(ref output[i.X], weights[i.X, i.Y] * neurons[i.Y]);


        public static Action<Index1D, ArrayView<float>, ArrayView<float>> Set;
        private static void SetKernel(Index1D i, ArrayView<float> biases, ArrayView<float> z)
            => z[i] = biases[i];




        #region Function Kernels

        public static Action<Index1D, ArrayView<float>, ArrayView<float>> eLU;
        private static void eLUKernel(Index1D i, ArrayView<float> data, ArrayView<float> output)
        {
            if (data[i] > 0)
                output[i] = data[i];
            else
                output[i] = (float)(0.1f * (XMath.Exp(data[i]) - 1));
        }


        public static Action<Index1D, ArrayView<float>, ArrayView<float>> eLUPrime;
        private static void eLUPrimeKernel(Index1D i, ArrayView<float> data, ArrayView<float> output)
        {
            if (data[i] > 0)
                output[i] = 1;
            else
                output[i] = (float)(0.1f * (XMath.Exp(data[i]) - 1)) + 0.1f;
        }


        public static Action<Index1D, ArrayView<float>, ArrayView<float>> Linear;
        public static Action<Index1D, ArrayView<float>, ArrayView<float>> LinearPrime;

        private static void Set1Kernel(Index1D i, ArrayView<float> data, ArrayView<float> output)
        {
            output[i] = 1;
        }




        public static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float, float> ModifyBiases;
        private static void ModifyBiasesKernel(Index1D i, ArrayView<float> moveBiases, ArrayView<float> biases, ArrayView<float> movingAverage, float inputsLength, float beta, float learningRate)
        {
            moveBiases[i] /= inputsLength;
            movingAverage[i] = beta * movingAverage[i] + (1 - beta) * moveBiases[i] * moveBiases[i];
            biases[i] -= moveBiases[i] * learningRate * XMath.Rsqrt(movingAverage[i]);
        }


        public static Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float, float, float> ModifyWeights;
        private static void ModifyWeightsKernel(Index2D i, ArrayView2D<float, Stride2D.DenseX> moveWeights, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> movingAverage, float inputsLength, float beta, float learningRate)
        {
            moveWeights[i.X, i.Y] /= inputsLength;
            movingAverage[i.X, i.Y] = beta * movingAverage[i.X, i.Y] + (1 - beta) * moveWeights[i.X, i.Y] * moveWeights[i.X, i.Y];
            weights[i.X, i.Y] -= moveWeights[i.X, i.Y] * learningRate * XMath.Rsqrt(movingAverage[i.X, i.Y]);
        }
        

        public static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> SetLastLayerError;
        private static void SetLastLayerErrorKernel(Index1D i, ArrayView<float> nnOutput, ArrayView<float> target, ArrayView<float> z, ArrayView<float> output)
        {
            /*if (z[i] <= 0)
                z[i] = (float)(0.1f * (XMath.Exp(z[i]) - 1));*/ //Linear Derivative
            z[i] = 1;


            output[i.X] = 2 * (nnOutput[i] - target[i]) * z[i];
        }

        public static Action<Index1D, ArrayView<float>, ArrayView<float>> MultZFunctionned;
        private static void MultZFunctionnedKernel(Index1D i, ArrayView<float> data, ArrayView<float> z)
        {
            if (z[i] > 0)
                z[i] = 1;
            else
                z[i] = (float)(0.1f * (XMath.Exp(z[i]) - 1)) + 0.1f;

            /*if (z[i] > 0)
                z[i] = 1;
            else
                z[i] = 0;*/

            data[i] = data[i] * z[i];
        }

        private static void MatrixVectorMultKernel(Index2D i, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView<float> vector, ArrayView<float> output)
        {
            Atomic.Add(ref output[i.X], matrix[i.X, i.Y] * vector[i.Y]);

            float[] final;
            for(int x = 0; x < i.X; x++)
            {
                //final += matrix[i.X, i.Y] + 
            }
        }

        #endregion
    }
}
