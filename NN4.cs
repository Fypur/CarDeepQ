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
    public class NN4
    {
        public Accelerator Accelerator => Kernels.Accelerator;

        public int[] Layers;

        public MemoryBuffer1D<float, Stride1D.Dense>[] Neurons;
        public MemoryBuffer1D<float, Stride1D.Dense>[] Z;

        public MemoryBuffer1D<float, Stride1D.Dense>[] Biases;
        public MemoryBuffer2D<float, Stride2D.DenseX>[] Weights;

        public MemoryBuffer2D<float, Stride2D.DenseX>[] MovingAverage;
        public MemoryBuffer1D<float, Stride1D.Dense>[] MovingAverageBiases;

        public MemoryBuffer1D<float, Stride1D.Dense>[] Intermediate;
        public MemoryBuffer1D<float, Stride1D.Dense>[] Error;
        public MemoryBuffer2D<float, Stride2D.DenseX>[] MoveWeights;
        public MemoryBuffer1D<float, Stride1D.Dense>[] MoveBiases;

        private MemoryBuffer1D<float, Stride1D.Dense> target;
        

        public float LearningRate;
        public static Action<Index1D, ArrayView<float>, ArrayView<float>> ActivationHidden = Kernels2.eLU;
        public static Action<Index1D, ArrayView<float>, ArrayView<float>> ActivationOut = Kernels2.Linear;

        public static Action<Index1D, ArrayView<float>, ArrayView<float>> ActivationHiddenDer = Derivatives(ActivationHidden);
        public static Action<Index1D, ArrayView<float>, ArrayView<float>> ActivationOutDer = Derivatives(ActivationOut);

        public float Beta = 0.9f;

        #region Init

        public NN4(int[] layers, float learningRate)
        {
            Setup(layers, learningRate, null, null);
        }

        public NN4(int[] layers, float learningRate, float[][][] weights, float[][] biases)
        {
            Setup(layers, learningRate, weights, biases);
        }

        public NN4(int[] layers, float learningRate, MemoryBuffer2D<float, Stride2D.DenseX>[] weights, MemoryBuffer1D<float, Stride1D.Dense>[] biases)
        {
            Layers = layers;
            LearningRate = learningRate;
            Weights = weights;
            Biases = biases;
            var param = ConvertToJaggedArray();
            Setup(layers, learningRate, param.Item1, param.Item2);
        }

        public void Setup(int[] layers, float learningRate, float[][][] weights, float[][] biases)
        {
            Layers = layers;
            LearningRate = learningRate;


            Neurons = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];
            Z = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];

            Biases = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];
            MovingAverageBiases = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];

            Weights = new MemoryBuffer2D<float, Stride2D.DenseX>[Layers.Length];
            MovingAverage = new MemoryBuffer2D<float, Stride2D.DenseX>[Layers.Length];

            Intermediate = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];
            Error = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];
            MoveBiases = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];
            MoveWeights = new MemoryBuffer2D<float, Stride2D.DenseX>[Layers.Length];

            Neurons[0] = Accelerator.Allocate1D(new float[Layers[0]]);
            Z[0] = Accelerator.Allocate1D(new float[Layers[0]]);

            target = Accelerator.Allocate1D(new float[Layers[Layers.Length - 1]]);

            bool randomW = weights == null;
            bool randomB = biases == null;

            for (int l = 1; l < Layers.Length; l++)
            {
                float std = (float)Math.Sqrt(2.0 / Layers[l - 1]);

                Neurons[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                Z[l] = Accelerator.Allocate1D(new float[Layers[l]]);

                Biases[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                MovingAverageBiases[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                Intermediate[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                Error[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                MoveBiases[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                MoveWeights[l] = Accelerator.Allocate2DDenseX(new float[Layers[l], Layers[l - 1]]);

                Weights[l] = Accelerator.Allocate2DDenseX(new float[Layers[l], Layers[l - 1]]);
                MovingAverage[l] = Accelerator.Allocate2DDenseX(new float[Layers[l], Layers[l - 1]]);

                float[] biasesN = new float[Layers[l]];
                float[,] weightsN = new float[Layers[l], Layers[l - 1]];

                //Obligé de faire ça par for loop car random utilisé
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    biasesN[n] = randomB ? GaussianRandom(0, 0.5f) : biases[l][n];

                    for (int prevLayerN = 0; prevLayerN < Neurons[l - 1].Length; prevLayerN++)
                        weightsN[n, prevLayerN] = randomW ? GaussianRandom(0, std) : weights[l][n][prevLayerN];
                }

                Biases[l].CopyFromCPU(biasesN);
                Weights[l].CopyFromCPU(weightsN);

                Kernels.SetValue((Index1D)MovingAverageBiases[l].Length, MovingAverageBiases[l].View, 1);
                Kernels.SetValue2D(MovingAverage[l].IntExtent, MovingAverage[l], 1);
            }
        }

        #endregion


        public float[] FeedForward(float[] input, bool getOutput = true)
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

            if(getOutput)
                return Neurons[Neurons.Length - 1].GetAsArray1D();
            return null;
        }

        public float[] FeedForward(MemoryBuffer1D<float, Stride1D.Dense> input, bool getOutput = true)
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
        }

        public void Train(MemoryBuffer1D<float, Stride1D.Dense>[] inputs, float[][] targets)
        {
            float totalCost = 0;
            for (int l = 1; l < Layers.Length; l++)
            {
                MoveBiases[l].MemSetToZero();
                MoveWeights[l].MemSetToZero();
            }

            for (int p = 0; p < inputs.Length; p++)
            {
                FeedForward(inputs[p], true);
                target.CopyFromCPU(targets[p]);

                int lastLayer = Layers.Length - 1;

                Kernels2.SetLastLayerError((int)Error[lastLayer].Length, Neurons[lastLayer].View, target.View, Z[lastLayer].View, Error[lastLayer].View);


                for (int l = Layers.Length - 1; l >= 2; l--)
                {
                    Error[l - 1].MemSetToZero();

                    Kernels.MatrixVectorMult2(new Index2D((int)Neurons[l].Length, (int)Neurons[l - 1].Length), Weights[l], Error[l].View, Error[l - 1].View);
                    Kernels2.MultZFunctionned((int)MoveBiases[l - 1].Length, Error[l - 1].View, Z[l - 1].View);

                    Kernels.VectorAdd(Layers[l], MoveBiases[l].View, Error[l].View, MoveBiases[l].View);
                    Kernels.MatrixSetVectorSqrdMult(new Index2D(MoveWeights[l].ElementSize, MoveWeights[l].ElementSize), MoveWeights[l].View, MoveBiases[l].View, Neurons[l - 1].View, MoveWeights[l].View);
                }

                Kernels.VectorAdd(Layers[1], MoveBiases[1].View, Error[1].View, MoveBiases[1].View);
                Kernels.MatrixSetVectorSqrdMult(new Index2D(MoveWeights[1].ElementSize, MoveWeights[1].ElementSize), MoveWeights[1].View, MoveBiases[1].View, Neurons[0].View, MoveWeights[1].View);
            }

            
            for (int l = 1; l < Layers.Length; l++)
            {
                Kernels2.ModifyBiases(Layers[l], MoveBiases[l].View, Biases[l].View, MovingAverageBiases[l].View, inputs.Length, Beta, LearningRate);
                Kernels2.ModifyWeights(new Index2D(Layers[l], Layers[l - 1]), MoveWeights[l].View, Weights[l].View, MovingAverage[l].View, inputs.Length, Beta, LearningRate);
            }

            totalCost = totalCost / inputs.Length;
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

        public void Save(string outputDir)
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
        }

        #endregion
    }
}
