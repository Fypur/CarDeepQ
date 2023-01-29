using CarDeepQ;
using Fiourp;
using FMOD;
using Microsoft.Xna.Framework.Content;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace CarDeepQ
{
    public class NN5
    {
        public Accelerator Accelerator => Kernels.Accelerator;

        public MemoryBuffer1D<int, Stride1D.Dense> Layers;
        public const int MaxLayerSize = 256;
        public const int LayerLength = 4;

        public MemoryBuffer3D<float, Stride3D.DenseXY> Weights;
        public MemoryBuffer2D<float, Stride2D.DenseX> Biases;

        public MemoryBuffer3D<float, Stride3D.DenseXY> MovingAverage;
        public MemoryBuffer2D<float, Stride2D.DenseX> MovingAverageBiases;

        public MemoryBuffer3D<float, Stride3D.DenseXY> MoveWeights;
        public MemoryBuffer2D<float, Stride2D.DenseX> MoveBiases;


        public float LearningRate;

        public float Beta = 0.9f;

        #region Init

        public NN5(int[] layers, float learningRate)
        {
            Setup(layers, learningRate, null, null);
        }

        public NN5(int[] layers, float learningRate, float[][][] weights, float[][] biases)
        {
            Setup(layers, learningRate, weights, biases);
        }

        /*public NN5(int[] layers, float learningRate, MemoryBuffer3D<float, Stride3D.DenseXY> weights, MemoryBuffer2D<float, Stride2D.DenseX> biases)
        {
            Layers = layers;
            LearningRate = learningRate;
            Weights = weights;
            Biases = biases;
            var param = ConvertToJaggedArray();
            Setup(layers, learningRate, param.Item1, param.Item2);
        }*/

        public void Setup(int[] layers, float learningRate, float[][][] weights, float[][] biases)
        {
            Layers = Accelerator.Allocate1D(layers);
            LearningRate = learningRate;

            bool randomW = weights == null;
            bool randomB = biases == null;


            /*MaxLayerSize = layers[0];

            for(int l = 1; l < layers.Length; l++)
                if(layers[l] > MaxLayerSize)
                    MaxLayerSize = layers[l];*/

            float[,,] w = new float[layers.Length, MaxLayerSize, MaxLayerSize];
            float[,] b = new float[layers.Length, MaxLayerSize];

            for (int l = 1; l < layers.Length; l++)
            {
                float std = (float)Math.Sqrt(2.0 / layers[l - 1]);

                for (int n = 0; n < layers[l]; n++)
                {
                    b[l, n] = randomB ? GaussianRandom(0, 0.5f) : biases[l][n];

                    for (int prevLayerN = 0; prevLayerN < layers[l - 1]; prevLayerN++)
                        w[l, n, prevLayerN] = randomW ? GaussianRandom(0, std) : weights[l][n][prevLayerN];
                }
            }

            Biases = Accelerator.Allocate2DDenseX(b);
            Weights = Accelerator.Allocate3DDenseXY(w);

            LongIndex2D lB = new LongIndex2D(LayerLength, MaxLayerSize);
            LongIndex3D lW = new LongIndex3D(LayerLength, MaxLayerSize, MaxLayerSize);

            MovingAverageBiases = Accelerator.Allocate2DDenseX(b);
            MovingAverage = Accelerator.Allocate3DDenseXY(w);

            MoveBiases = Accelerator.Allocate2DDenseX(b);
            MoveWeights = Accelerator.Allocate3DDenseXY(w);

            Kernels.SetValue3D((Index3D)MovingAverage.Extent, MovingAverage.View, 1);
            Kernels.SetValue2D((Index2D)MovingAverageBiases.Extent, MovingAverageBiases.View, 1);
        }

        #endregion


        /*public float[] FeedForward(float[] input, bool getOutput = true)
        {
            Neurons[0].CopyFromCPU(input);

            for (int l = 1; l < Layers.Length; l++)
            {
                Z[l].View.CopyFrom(Biases[l].View);
                Kernels2.ForwardLayer(new Index2D(Layers[l], Layers[l - 1]), Weights[l], Neurons[l - 1].View, Z[l].View);

                if (l != Layers.Length - 1)
                    ActivationHidden((int)Neurons[l].Length, Z[l].View, Neurons[l].View);
                else
                    ActivationOut((int)Neurons[l].Length, Z[l].View, Neurons[l].View);
            }

            if (getOutput)
                return Neurons[Neurons.Length - 1].GetAsArray1D();
            return null;
        }*/

        /*public float[] FeedForward(MemoryBuffer1D<float, Stride1D.Dense> input, bool getOutput = true)
        {
            Neurons[0].CopyFrom(input);

            for (int l = 1; l < Layers.Length; l++)
            {
                Z[l].View.CopyFrom(Biases[l].View);
                Kernels2.ForwardLayer(new Index2D(Layers[l], Layers[l - 1]), Weights[l], Neurons[l - 1].View, Z[l].View);

                if (l != Layers.Length - 1)
                    ActivationHidden((int)Neurons[l].Length, Z[l].View, Neurons[l].View);
                else
                    ActivationOut((int)Neurons[l].Length, Z[l].View, Neurons[l].View);
            }

            if (getOutput)
                return Neurons[Neurons.Length - 1].GetAsArray1D();
            //return new float[Neurons[Neurons.Length - 1].Length];
            return null;
        }*/

        public void Train(MemoryBuffer1D<float, Stride1D.Dense>[] inputs, float[,] targets)
        {
            /////////////////////////////////////////////////////////////////////////////////////ZROIHZPROUGHZPRU9OIHZOURFHZOURG NOT WORKING YET INPUTS ARE BAD
            MemoryBuffer2D<float, Stride2D.DenseX> inp = Accelerator.Allocate2DDenseX(targets);

            MoveBiases.MemSetToZero();
            MoveWeights.MemSetToZero();

            MemoryBuffer2D<float, Stride2D.DenseX> targ = Accelerator.Allocate2DDenseX(targets);

            Kernels2.BackPropInputs(inputs.Length, inp.View, targ.View, Layers.View, MaxLayerSize, Weights.View, Biases.View, MoveWeights.View, MoveBiases.View);

            Kernels2.MoveParameters(Layers.IntExtent - 2, Layers.View, LearningRate, Beta, inputs.Length, MoveWeights.View, MoveBiases.View, Weights.View, Biases.View, MovingAverage.View, MovingAverageBiases.View);
        }

        /*public void CheckNetwork()
        {
            for (int l = 0; l < Layers.Length; l++)
            {
                Check(Neurons[l]);

                if (l != 0)
                {
                    Check(Biases[l]);

                    for (int n = 0; n < Neurons[l].Length; n++)
                        Check(Weights[l][n]);
                }
            }
        }*/

        #region Activations

        public static Action<Index1D, ArrayView<float>, ArrayView<float>> Derivatives(Action<Index1D, ArrayView<float>, ArrayView<float>> function)
        {
            if (function == Kernels.VectorSigmoid)
                return Kernels.VectorSigmoidPrime;
            if (function == Kernels.VectorReLU)
                return Kernels.VectorReLUPrime;
            if (function == Kernels2.eLU)
                return Kernels2.eLUPrime;
            if (function == Kernels2.Linear)
                return Kernels2.LinearPrime;

            throw new Exception("Could not find derivative of Activation Function");
        }

        #endregion

        #region Checks

        void Check(float[] f)
        {
            for (int i = 0; i < f.Length; i++)
                Check(f[i]);
        }

        void Check(float f)
        {
            if (f == float.NaN || f > 10000)
                throw new Exception("float has been checked as NaN or too big");
        }

        #endregion

        #region Utils

        //https://stackoverflow.com/questions/218060/random-gaussian-variables
        private float GaussianRandom()
        {
            double u1 = 1.0 - Rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - Rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return (float)randStdNormal;
        }

        private float GaussianRandom(float mean, float standardDeviation)
        {
            double u1 = 1.0 - Rand.NextDouble(); //uniform(0,1] random doubles/home/f/Documents/CarDeepQ/saves/netweights
            double u2 = 1.0 - Rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = mean + standardDeviation * randStdNormal; //random normal(mean,stdDev^2)
            return (float)randNormal;
        }

        /*public void Save(string outputDir)
        {
            if (Directory.Exists(outputDir.Substring(0, outputDir.Substring(0, outputDir.Length - 2).LastIndexOf('\\'))))
                Directory.CreateDirectory(outputDir);
            else
                throw new Exception("Parent Dir does not exist");

            var param = ConvertToJaggedArray();
            float[][][] w = param.Item1;
            float[][] b = param.Item2;

            string jsonW = JsonSerializer.Serialize(w);
            string jsonB = JsonSerializer.Serialize(b);
            File.WriteAllText(outputDir + "weights", jsonW);
            File.WriteAllText(outputDir + "biases", jsonB);
        }

        public void Load(string inputDir)
        {
            string jsonW = File.ReadAllText(inputDir + "weights");
            string jsonB = File.ReadAllText(inputDir + "biases");

            float[][][] w = JsonSerializer.Deserialize<float[][][]>(jsonW);
            float[][] b = JsonSerializer.Deserialize<float[][]>(jsonB);

            for (int l = 1; l < Layers.Length; l++)
            {
                Biases[l].CopyFromCPU(b[l]);

                float[,] w2 = new float[Layers[l], Layers[l - 1]];
                for (int n = 0; n < Layers[l]; n++)
                    for (int prevN = 0; prevN < Layers[l - 1]; prevN++)
                        w2[n, prevN] = w[l][n][prevN];

                Weights[l].CopyFromCPU(w2);
            }
        }

        public NN4 Copy()
        {
            NN4 n = new NN4(Layers, LearningRate, Weights, Biases);
            return n;
        }

        public Tuple<float[][][], float[][]> ConvertToJaggedArray()
        {
            float[][][] w = new float[Weights.Length][][];
            float[][] b = new float[Biases.Length][];

            for (int l = 1; l < Layers.Length; l++)
            {
                b[l] = new float[Biases[l].Length];
                Biases[l].CopyToCPU(b[l]);

                float[,] w2 = new float[Layers[l], Layers[l - 1]];
                Weights[l].CopyToCPU(w2);

                w[l] = new float[Layers[l]][];
                for (int n = 0; n < Layers[l]; n++)
                {
                    w[l][n] = new float[Layers[l - 1]];
                    for (int prevN = 0; prevN < Layers[l - 1]; prevN++)
                        w[l][n][prevN] = w2[n, prevN];

                }
            }

            return new Tuple<float[][][], float[][]>(w, b);
        }*/

        #endregion
    }
}
