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

namespace CarDeepQ
{
    public class NN2
    {
        public int[] Layers;

        public float[][] Neurons;
        public float[][] Z;
        public float[][] Biases;
        public float[][][] Weights;
        public float[][][] MovingAverage;
        public float[][] MovingAverageBiases;

        public float LearningRate;
        public static Func<float, float> ActivationHidden = ReLU;
        public static Func<float, float> ActivationOut = ReLU;

        public static Func<float, float> ActivationHiddenDer = Derivatives(ActivationHidden);
        public static Func<float, float> ActivationOutDer = Derivatives(ActivationOut);

        public float Beta = 0.9f;

        public NN2(int[] layers, float learningRate)
        {
            Layers = layers;
            LearningRate = learningRate;

            //Init Everything
            Neurons = new float[Layers.Length][];
            Z = new float[Layers.Length][];
            Biases = new float[Layers.Length][];
            MovingAverageBiases = new float[Layers.Length][];
            Weights = new float[Layers.Length][][];
            MovingAverage = new float[Layers.Length][][];

            Neurons[0] = new float[Layers[0]];


            for (int l = 1; l < Layers.Length; l++)
            {
                Neurons[l] = new float[Layers[l]];
                Z[l] = new float[Layers[l]];
                Biases[l] = new float[Layers[l]];
                MovingAverageBiases[l] = new float[Layers[l]];
                Weights[l] = new float[Layers[l]][];
                MovingAverage[l] = new float[Layers[l]][];

                for(int n = 0; n < Neurons[l].Length; n++)
                {
                    Biases[l][n] = GaussianRandom(0, 0.5f);
                    MovingAverageBiases[l][n] = 1;
                    Weights[l][n] = new float[Layers[l - 1]];
                    MovingAverage[l][n] = new float[Layers[l - 1]];
                    float std = (float)Math.Sqrt(2.0 / Layers[l - 1]);

                    for (int prevLayerN = 0; prevLayerN < Neurons[l - 1].Length; prevLayerN++)
                    {
                        Weights[l][n][prevLayerN] = GaussianRandom(0, std);
                        MovingAverage[l][n][prevLayerN] = 1;
                    }
                }
            }
        }


        public float[] FeedForward(float[] input)
        {
            if (input.Length != Layers[0])
                throw new Exception("Input is not of right size");

            if (input.Contains(float.NaN))
                throw new Exception("Input contains NaN values");

            Neurons[0] = (float[])input.Clone();

            for(int l = 1; l < Layers.Length; l++)
            {
                for(int n = 0; n < Neurons[l].Length; n++)
                {
                    Z[l][n] = 0;

                    for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)
                        Z[l][n] += Weights[l][n][prevN] * Neurons[l - 1][prevN];

                    Z[l][n] += Biases[l][n];


                    if(l != Layers.Length - 1)
                        Neurons[l][n] = ActivationHidden(Z[l][n]);
                    else
                        Neurons[l][n] = ActivationOut(Z[l][n]);
                }
            }

            return (float[])Neurons[Layers.Length - 1].Clone();
        }


        public void Train(float[][] inputs, float[][] targets)
        {
            float[][] moveBiases = new float[Biases.Length][];
            float[][][] moveWeights = new float[Weights.Length][][];
            float[][] error = new float[Layers.Length][];

            for (int l = 1; l < Layers.Length; l++)
            {
                error[l] = new float[Neurons[l].Length];
                moveBiases[l] = new float[Biases[l].Length];
                moveWeights[l] = new float[Weights[l].Length][];

                for (int i = 0; i < Layers[l]; i++)
                    moveWeights[l][i] = new float[Weights[l][i].Length];
            }

            float totalCost = 0;
            for (int p = 0; p < inputs.Length; p++)
            {
                float[] input = inputs[p];
                float[] target = targets[p];
                float[] output = FeedForward(input);

                //Computing the error
                //The error is basically the derivative of the cost by the z of that neuron at that place
                for (int i = 0; i < Layers[Layers.Length - 1]; i++)
                    error[Neurons.Length - 1][i] = 2 * (output[i] - target[i]) * ActivationOutDer(Z[Layers.Length - 1][i]);

                for(int l = Layers.Length - 1; l >= 2; l--)
                {
                    error[l - 1] = new float[Neurons[l - 1].Length];
                    for(int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)
                    {
                        for (int n = 0; n < Neurons[l].Length; n++)
                            error[l - 1][prevN] += error[l][n] * Weights[l][n][prevN];

                        error[l - 1][prevN] *= ActivationHiddenDer(Z[l - 1][prevN]);
                    }
                }


                for (int l = 1; l < Layers.Length; l++)
                {
                    for (int n = 0; n < Neurons[l].Length; n++)
                    {
                        moveBiases[l][n] -= error[l][n];

                        for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)
                            moveWeights[l][n][prevN] -= error[l][n] * Neurons[l - 1][prevN];
                    }
                }

                #region cost and plotting
                float cost = 0;
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
                Console.ForegroundColor = ConsoleColor.Gray;
                #endregion
            }


            for (int l = 1; l < Layers.Length; l++)
            {
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    moveBiases[l][n] /= inputs.Length;
                    MovingAverageBiases[l][n] = Beta * MovingAverageBiases[l][n] + (1 - Beta) * moveBiases[l][n] * moveBiases[l][n];
                    Biases[l][n] += moveBiases[l][n] * (LearningRate / (float)Math.Sqrt(MovingAverageBiases[l][n]));
                    for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)
                    {
                        moveWeights[l][n][prevN] /= inputs.Length;
                        MovingAverage[l][n][prevN] = Beta * MovingAverage[l][n][prevN] + (1 - Beta) * moveWeights[l][n][prevN] * moveWeights[l][n][prevN];
                        Weights[l][n][prevN] += moveWeights[l][n][prevN] * (LearningRate / (float)Math.Sqrt(MovingAverage[l][n][prevN]));
                    }
                }
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

        public void CheckNetwork()
        {
            for (int l = 0; l < Layers.Length; l++)
            {
                Check(Neurons[l]);

                if(l != 0)
                {
                    Check(Biases[l]);

                    for (int n = 0; n < Neurons[l].Length; n++)
                        Check(Weights[l][n]);
                }
            }
        }

        #region Activations
        public static float Sigmoid(float x)
            => (float)(1 / (1 + Math.Exp(-x)));

        private static float SigmoidPrime(float x)
            => Sigmoid(x) * (1 - Sigmoid(x));

        private static float ReLU(float x)
        {
            if (x >= 0)
                return x;
            return 0;
        }

        private static float ReLUPrime(float x)
        {
            if (x > 0)
                return 1;
            return 0;
        }

        private static float eLU(float x)
        {
            if (x > 0)
                return x;
            return (float)(0.1f * (Math.Exp(x) - 1));
        }

        private static float eLUPrime(float x)
        {
            if (x > 0)
                return 1;
            return eLU(x) + 0.1f;
        }

        private static float Linear(float x)
            => x;

        private static float LinearPrime(float x)
            => 1;

        public static Func<float, float> Derivatives(Func<float, float> function)
        {
            if (function == Sigmoid)
                return SigmoidPrime;
            if (function == ReLU)
                return ReLUPrime;
            if (function == eLU)
                return eLUPrime;
            if (function == Linear)
                return LinearPrime;

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

            Weights = JsonSerializer.Deserialize<float[][][]>(jsonW);
            Biases = JsonSerializer.Deserialize<float[][]>(jsonB);
        }

        public NN2 Copy()
        {
            NN2 n = new NN2(Layers, LearningRate) { Weights = (float[][][])this.Weights.Clone(), Biases = (float[][])this.Biases.Clone() };
            return n;
        }

        #endregion
    }
}
