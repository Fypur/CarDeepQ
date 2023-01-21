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

namespace CarDeepQ
{
    public class NN3
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

        public float LearningRate;
        public static Action<Index1D, ArrayView<float>, ArrayView<float>> ActivationHidden = Kernels.VectoreLU;
        public static Action<Index1D, ArrayView<float>, ArrayView<float>> ActivationOut = Kernels.VectorLinear;

        public static Action<Index1D, ArrayView<float>, ArrayView<float>> ActivationHiddenDer = Derivatives(ActivationHidden);
        public static Action<Index1D, ArrayView<float>, ArrayView<float>> ActivationOutDer = Derivatives(ActivationOut);

        public float Beta = 0.9f;

        public NN3(int[] layers, float learningRate)
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

            Neurons[0] = Accelerator.Allocate1D(new float[Layers[0]]);
            Z[0] = Accelerator.Allocate1D(new float[Layers[0]]);

            
            for (int l = 1; l < Layers.Length; l++)
            {
                float std = (float)Math.Sqrt(2.0 / Layers[l - 1]);

                Neurons[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                Z[l] = Accelerator.Allocate1D(new float[Layers[l]]);

                Biases[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                MovingAverageBiases[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                Intermediate[l] = Accelerator.Allocate1D(new float[Layers[l]]);

                Weights[l] = Accelerator.Allocate2DDenseX(new float[Layers[l], Layers[l - 1]]);
                MovingAverage[l] = Accelerator.Allocate2DDenseX(new float[Layers[l], Layers[l - 1]]);


                float[] biasesN = new float[Layers[l]];
                float[,] weightsN = new float[Layers[l], Layers[l - 1]];

                //Obligé de faire ça par for loop car random utilisé
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    biasesN[n] = GaussianRandom(0, 0.5f);

                    for (int prevLayerN = 0; prevLayerN < Neurons[l - 1].Length; prevLayerN++)
                        weightsN[n, prevLayerN] = GaussianRandom(0, std);
                }

                Biases[l].CopyFromCPU(biasesN);
                Weights[l].CopyFromCPU(weightsN);

                Kernels.SetValue((Index1D)MovingAverageBiases[l].Length, MovingAverageBiases[l].View, 1);
                Kernels.SetValue2D(new Index2D(MovingAverage[l].ElementSize, MovingAverage[l].ElementSize), MovingAverage[l], 1);
            }
        }

        public NN3(int[] layers, float learningRate, float[][][] weights, float[][] biases)
        {
            Layers = layers;
            LearningRate = learningRate;

            //Init Everything
            Neurons = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];
            Z = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];
            Biases = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];
            MovingAverageBiases = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];
            Intermediate = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];

            Weights = new MemoryBuffer2D<float, Stride2D.DenseX>[Layers.Length];
            MovingAverage = new MemoryBuffer2D<float, Stride2D.DenseX>[Layers.Length];

            Neurons[0] = Accelerator.Allocate1D(new float[Layers[0]]);
            Z[0] = Accelerator.Allocate1D(new float[Layers[0]]);

            for (int l = 1; l < Layers.Length; l++)
            {
                float std = (float)Math.Sqrt(2.0 / Layers[l - 1]);

                Neurons[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                Z[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                Biases[l] = Accelerator.Allocate1D(new float[Layers[l]]);

                Intermediate[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                MovingAverageBiases[l] = Accelerator.Allocate1D(new float[Layers[l]]);

                Weights[l] = Accelerator.Allocate2DDenseX(new float[Layers[l], Layers[l - 1]]);
                MovingAverage[l] = Accelerator.Allocate2DDenseX(new float[Layers[l], Layers[l - 1]]);

                float[] biasesN = new float[Layers[l]];
                float[,] weightsN = new float[Layers[l], Layers[l - 1]];

                //Obligé de faire ça par for loop car random utilisé
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    biasesN[n] = biases[l][n];

                    for (int prevLayerN = 0; prevLayerN < Neurons[l - 1].Length; prevLayerN++)
                        weightsN[n, prevLayerN] = weights[l][n][prevLayerN];
                }

                Biases[l].CopyFromCPU(biasesN);
                Weights[l].CopyFromCPU(weightsN);

                Kernels.SetValue((Index1D)MovingAverageBiases[l].Length, MovingAverageBiases[l].View, 1);
                Kernels.SetValue2D(new Index2D(MovingAverage[l].ElementSize, MovingAverage[l].ElementSize), MovingAverage[l], 1);
            }
        }


        public float[] FeedForward(float[] input)
        {
            if (input.Length != Layers[0])
                throw new Exception("Input is not of right size");

            if (input.Contains(float.NaN))
                throw new Exception("Input contains NaN values");

            Neurons[0].CopyFromCPU(input);

            for (int l = 1; l < Layers.Length; l++)
            {
                Z[l].MemSetToZero();
                Kernels.SetValue((int)Z[l].Length, Z[l].View, 0);
                Kernels.MatrixVectorMult(new Index2D((int)Neurons[l].Length, (int)Neurons[l - 1].Length), Weights[l], Neurons[l - 1].View, Z[l].View);
                Kernels.VectorAdd(Layers[l], Z[l].View, Biases[l].View, Z[l].View);

                if (l != Layers.Length - 1)
                    ActivationHidden((int)Neurons[l].Length, Z[l].View, Neurons[l].View);
                else
                    ActivationOut((int)Neurons[l].Length, Z[l].View, Neurons[l].View);
            }

            return Neurons[Neurons.Length - 1].GetAsArray1D();
        }

        
        public void Train(float[][] inputs, float[][] targets)
        {
            MemoryBuffer1D<float, Stride1D.Dense>[] moveBiases = new MemoryBuffer1D<float, Stride1D.Dense>[Biases.Length];
            MemoryBuffer1D<float, Stride1D.Dense>[] error = new MemoryBuffer1D<float, Stride1D.Dense>[Layers.Length];

            MemoryBuffer2D<float, Stride2D.DenseX>[] moveWeights = new MemoryBuffer2D<float, Stride2D.DenseX>[Weights.Length];

            for (int l = 1; l < Layers.Length; l++)
            {
                error[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                moveBiases[l] = Accelerator.Allocate1D(new float[Layers[l]]);
                moveWeights[l] = Accelerator.Allocate2DDenseX(new float[Layers[l], Layers[l - 1]]);
            }

            float totalCost = 0;
            for (int p = 0; p < inputs.Length; p++)
            {
                FeedForward(inputs[p]);
                MemoryBuffer1D<float, Stride1D.Dense> target = Accelerator.Allocate1D(targets[p]);

                //Computing the error
                //The error is basically the derivative of the cost by the z of that neuron at that place


                int lastLayer = Layers.Length - 1;
                Kernels.VectorSub((int)error[lastLayer].Length, Neurons[lastLayer].View, target.View, error[lastLayer].View);
                Kernels.VectorMultConstant((int)error[lastLayer].Length, error[lastLayer].View, 2, error[lastLayer].View);

                ActivationOutDer((int)Z[lastLayer - 1].Length, Z[lastLayer].View, Intermediate[lastLayer].View);
                Kernels.VectorMult((int)error[lastLayer].Length, error[lastLayer].View, Intermediate[lastLayer].View, error[lastLayer].View);


                for (int l = Layers.Length - 1; l >= 2; l--)
                {
                    Kernels.MatrixVectorMult(new Index2D((int)Neurons[l].Length, (int)Neurons[l - 1].Length), Weights[l], error[l].View, error[l - 1].View);

                    ActivationHiddenDer((int)Z[l - 1].Length, Z[l - 1].View, Intermediate[l - 1].View);
                    Kernels.VectorMult((int)error[l - 1].Length, error[l - 1].View, Intermediate[l - 1].View, error[l - 1].View);


                    /*for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)
                    {
                        for (int n = 0; n < Neurons[l].Length; n++)
                            error[l - 1][prevN] += error[l][n] * Weights[l][n][prevN];

                        error[l - 1][prevN] *= ActivationHiddenDer(Z[l - 1][prevN]);
                    }*/
                }


                for (int l = 1; l < Layers.Length; l++)
                {
                    Kernels.VectorAdd((int)moveBiases[l].Length, moveBiases[l].View, error[l].View, moveBiases[l].View);
                    Kernels.Matrix2VectorMult(new Index2D(moveWeights[l].ElementSize, moveWeights[l].ElementSize), moveWeights[l].View, error[l].View, Neurons[l - 1].View, moveWeights[l].View);

                    //Kernels.VectorMult((int)error[l].Length, error[l].View, Neurons[l - 1].View, error[l].View);

                    /*for (int n = 0; n < Neurons[l].Length; n++)
                    {
                        moveBiases[l][n] -= error[l][n];

                        for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)
                            moveWeights[l][n][prevN] -= error[l][n] * Neurons[l - 1][prevN];

                        //m[i.X, i.Y] += v1[i.X] * v2[i.Y];
                    }*/
                }

                #region cost and plotting
                /*float cost = 0;
                for (int i = 0; i < output.Length; i++)
                    cost += (float)Math.Pow(output[i] - target[i], 2);

                totalCost += cost;

                int good = 0;
                for (int i = 0; i < target.Length; i++)
                    if (target[i] == 1)
                    {
                        good = i;
                        break;
                    }

                float found = 0;
                int foundi = 0;
                for (int i = 0; i < output.Length; i++)
                    if (found < output[i])
                    {
                        found = output[i];
                        foundi = i;
                    }

                if (foundi == good)
                    Console.ForegroundColor = ConsoleColor.Green;
                else
                    Console.ForegroundColor = ConsoleColor.Red;
                //Console.WriteLine("cost: " + cost);
                Console.ForegroundColor = ConsoleColor.Gray;*/
                #endregion
            }


            for (int l = 1; l < Layers.Length; l++)
            {
                Kernels.VectorDivConstant((int)moveBiases[l].Length, moveBiases[l].View, inputs.Length, moveBiases[l].View);
                Kernels.MatrixDivConst(new Index2D(moveWeights[l].ElementSize, moveWeights[l].ElementSize), moveWeights[l].View, inputs.Length, moveWeights[l].View);

                Kernels.SetMovingAverageBiases((int)MovingAverageBiases[l].Length, MovingAverageBiases[l].View, moveBiases[l].View, Beta, MovingAverageBiases[l].View);
                Kernels.SetBiasesTrain((int)Biases[l].Length, Biases[l].View, moveBiases[l].View, MovingAverageBiases[l].View, LearningRate, Biases[l].View);

                Kernels.SetMovingAverage(new Index2D(MovingAverage[l].ElementSize, MovingAverage[l].ElementSize), MovingAverage[l].View, moveWeights[l].View, Beta, MovingAverage[l].View);
                Kernels.SetWeightsTrain(new Index2D(Weights[l].ElementSize, Weights[l].ElementSize), Weights[l].View, moveWeights[l].View, MovingAverage[l].View, LearningRate, Weights[l].View);


                /*//MovingAverageBiases[l][n] = Beta * MovingAverageBiases[l][n] + (1 - Beta) * moveBiases[l][n] * moveBiases[l][n];
                Kernels.VectorMult((int)moveBiases[l].Length, moveBiases[l].View, moveBiases[l].View, Intermediate[l].View);
                Kernels.VectorMultConstant((int)Intermediate[l].Length, Intermediate[l].View, 1 - Beta, Intermediate[l].View);
                Kernels.VectorMultConstant((int)MovingAverageBiases[l].Length, MovingAverageBiases[l].View, Beta, MovingAverageBiases[l].View);
                Kernels.VectorAdd((int)MovingAverageBiases[l].Length, MovingAverageBiases[l].View, Intermediate[l].View, MovingAverageBiases[l].View);

                //Biases[l][n] -= moveBiases[l][n] * (LearningRate / (float)Math.Sqrt(MovingAverageBiases[l][n]));
                Kernels.VectorInverseSqrt((int)MovingAverageBiases[l].Length, MovingAverageBiases[l].View, Intermediate[l].View);
                Kernels.VectorMultConstant((int)Intermediate[l].Length, Intermediate[l].View, -LearningRate, Intermediate[l].View);
                Kernels.VectorMult((int)moveBiases[l].Length, moveBiases[l].View, Intermediate[l].View, Intermediate[l].View);
                Kernels.VectorAdd((int)Biases[l].Length, Biases[l].View, Intermediate[l].View, Biases[l].View);*/
            }

            totalCost = totalCost / inputs.Length;
            //Console.WriteLine("total Cost : " + totalCost);
            /*if (totalCost < 1)
            {
                Main.pointsX.Add(Main.plotDist);
                Main.pointsY.Add(totalCost);
            }
            else
            {
                Main.pointsX.Add(Main.plotDist);
                Main.pointsY.Add(1);
            }

            Main.plotDist += 1;*/

            //CheckNetwork();
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
            if (function == Kernels.VectoreLU)
                return Kernels.VectoreLUPrime;
            if (function == Kernels.VectorLinear)
                return Kernels.VectorLinearPrime;

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

            string jsonW = JsonSerializer.Serialize(this.Weights);
            string jsonB = JsonSerializer.Serialize(this.Biases);
            File.WriteAllText(outputDir + "weights", jsonW);
            File.WriteAllText(outputDir + "biases", jsonB);
        }

        public void Load(string inputDir)
        {
            string jsonW = File.ReadAllText(inputDir + "weights");
            string jsonB = File.ReadAllText(inputDir + "biases");
            /*NeuralNetwork n = JsonSerializer.Deserialize<NeuralNetwork>(json);

            LearningRate = n.LearningRate;
            weights = n.weights;
            biases = n.biases;*/

            /*Weights = JsonSerializer.Deserialize<float[][][]>(jsonW);
            Biases = JsonSerializer.Deserialize<float[][]>(jsonB);*/
        }

        public NN2 Copy()
        {
            NN2 n = new NN2(Layers, LearningRate) { Weights = (float[][][])this.Weights.Clone(), Biases = (float[][])this.Biases.Clone() };
            return n;
        }

        #endregion
    }
}
