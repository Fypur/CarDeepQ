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
            Accelerator = Context.GetPreferredDevice(preferCPU:true).CreateAccelerator(Context);
            

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


            BackPropInputs = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView<int>, int, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>>(BackPropInputsKernel);

            MoveParameters = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, float, float, int, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>>(MoveParametersKernel);


            Accelerator.Synchronize();
        }




        public static Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>> ForwardLayer;
        private static void ForwardLayerKernel(Index2D i, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView<float> neurons, ArrayView<float> output)
            => Atomic.Add(ref output[i.X], weights[i.X, i.Y] * neurons[i.Y]);


        public static Action<Index1D, ArrayView<float>, ArrayView<float>> Set;
        private static void SetKernel(Index1D i, ArrayView<float> biases, ArrayView<float> z)
            => z[i] = biases[i];




        public static Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView<int>, int, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>> BackPropInputs;
        private static void BackPropInputsKernel(Index1D i, ArrayView2D<float, Stride2D.DenseX> inputs, ArrayView2D<float, Stride2D.DenseX> targets, ArrayView<int> layers, int maxLayerSize, ArrayView3D<float, Stride3D.DenseXY> weights, ArrayView2D<float, Stride2D.DenseX> biases, ArrayView3D<float, Stride3D.DenseXY> moveWeights, ArrayView2D<float, Stride2D.DenseX> moveBiases)
        {
            #region FeedForward

            float[,] Neurons = new float[NN5.LayerLength, NN5.MaxLayerSize];
            float[,] Z = new float[NN5.LayerLength, NN5.MaxLayerSize];

            for (int n = 0; n < layers[1]; n++)
            {
                Z[0, n] = 0;

                for (int prevN = 0; prevN < layers[0]; prevN++)
                    Z[0, n] = Z[0, n] + weights[1, n, prevN] * inputs[0, prevN];


                Z[0, n] = Z[0, n] + biases[1, n];

                //Neurons[1][n] = ActivationHidden(Z[l][n]);
                if (Z[0, n] > 0)
                    Neurons[1, n] = Z[0 ,n];
                else
                    Neurons[1, n] = (float)(0.1f * (XMath.Exp(Z[0, n]) - 1));
            }

            for (int l = 2; l < layers.Length; l++)
            {
                for (int n = 0; n < layers[l]; n++)
                {
                    Z[l, n] = 0;

                    for (int prevN = 0; prevN < layers[l - 1]; prevN++)
                        Z[l, n] = Z[0, n] + weights[l, n, prevN] * Neurons[l - 1, prevN];

                    Z[l, n] = Z[0, n] + biases[l, n];


                    if (l != layers.Length - 1)
                    {
                        //eLU
                        if (Z[0, n] > 0)
                            Neurons[1, n] = Z[0, n];
                        else
                            Neurons[1, n] = (float)(0.1f * (XMath.Exp(Z[0, n]) - 1));
                    }
                    else
                        Neurons[l, n] = Z[l, n]; //Linear
                }
            }

            #endregion

            /*#region BackPropInput

            float[,] moveBiasesInput = new float[NN5.LayerLength, NN5.MaxLayerSize];
            float[,,] moveWeightsInput = new float[NN5.LayerLength, NN5.MaxLayerSize, NN5.MaxLayerSize];
            float[,] error = new float[NN5.LayerLength, NN5.MaxLayerSize];

            for (int n = 0; n < layers[layers.Length - 1]; n++)
                error[Neurons.Length - 1, n] = 2 * (targets[i, n] - Neurons[layers[layers.Length - 1], n]) * 1; // * ActivationOutDer(Z[Layers.Length - 1][n]); LINEAR DERIVATIVE


            for (int l = layers.IntLength - 1; l >= 2; l--)
            {
                for (int prevN = 0; prevN < layers[l - 1]; prevN++)
                {
                    for (int n = 0; n < layers[l]; n++)
                        error[l - 1, prevN] = error[l - 1, prevN] + error[l, n] * weights[l, n, prevN];

                    //ActivationHiddenDerivative
                    if (Z[l - 1, prevN] > 0)
                        Z[l - 1, prevN] = 1;
                    else
                        Z[l - 1, prevN] = (float)(0.1f * (XMath.Exp(Z[l - 1, prevN]) - 1)) + 0.1f;

                    error[l - 1, prevN] = error[l - 1, prevN] * Z[l - 1, prevN];
                }

                for (int n = 0; n < layers[l]; n++)
                {
                    moveBiasesInput[l, n] = moveBiasesInput[l, n] + error[l, n];

                    for (int prevN = 0; prevN < layers[l - 1]; prevN++)
                        moveWeightsInput[l, n, prevN] = moveWeightsInput[l, n, prevN] + error[l, n] * Neurons[l - 1, prevN];
                }
            }

            for (int n = 1; n < layers[1]; n++)
            {
                moveBiasesInput[1, n] = moveBiasesInput[1, n] + error[1, n];

                for (int prevN = 1; prevN < layers[0]; prevN++)
                    moveWeightsInput[1, n, prevN] = moveWeightsInput[1, n, prevN] + error[1, n] * Neurons[0, prevN];
            }

            #endregion*/

            #region Sync All inputs

            /*for (int l = 1; l < layers.IntLength - 1; l++)
            {
                for (int n = 0; n < layers[l]; n++)
                {
                    Atomic.Add(ref moveBiases[l, n], moveBiasesInput[l, n]);

                    for (int prevN = 0; prevN < layers[l - 1]; prevN++)
                    {
                        Atomic.Add(ref moveWeights[l, n, prevN], moveWeightsInput[l, n, prevN]);
                    }
                }
            }*/

            #endregion
        }


        public static Action<Index1D, ArrayView<int>, float, float, int, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>> MoveParameters;
        private static void MoveParametersKernel(Index1D i, ArrayView<int> layers, float learningRate, float beta, int inputsLength, ArrayView3D<float, Stride3D.DenseXY> moveWeights, ArrayView2D<float, Stride2D.DenseX> moveBiases, ArrayView3D<float, Stride3D.DenseXY> weights, ArrayView2D<float, Stride2D.DenseX> biases, ArrayView3D<float, Stride3D.DenseXY> movingAverage, ArrayView2D<float, Stride2D.DenseX> movingAverageBiases)
        {
            i += 1;

            for (int n = 0; n < layers[i]; n++)
            {
                moveBiases[i, n] /= inputsLength;

                movingAverageBiases[i, n] = beta * movingAverageBiases[i, n] + (1 - beta) * moveBiases[i, n] * moveBiases[i, n];
                biases[i, n] += moveBiases[i, n] * learningRate * XMath.Rsqrt(movingAverageBiases[i, n]);

                for (int prevN = 0; prevN < layers[i - 1]; prevN++)
                {
                    moveWeights[i, n, prevN] /= inputsLength;

                    movingAverage[i, n, prevN] = beta * movingAverage[i, n, prevN] + (1 - beta) * moveWeights[i, n, prevN] * moveWeights[i, n, prevN];
                    weights[i, n, prevN] += moveWeights[i, n, prevN] * learningRate * XMath.Rsqrt(movingAverage[i, n, prevN]);
                }
            }
        }


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
