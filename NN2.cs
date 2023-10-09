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
        public static Func<float, float> ActivationHidden = eLU;
        public static Func<float, float> ActivationOut = Linear;

        public static Func<float, float> ActivationHiddenDer = Derivatives(ActivationHidden);
        public static Func<float, float> ActivationOutDer = Derivatives(ActivationOut);

        public float Beta = 0.9f;

        private float[][] error;
        private float[][][] moveWeights;
        private float[][] moveBiases;

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
            error = new float[Layers.Length][];

            Neurons[0] = new float[Layers[0]];
            Z[0] = new float[Layers[0]];
            error[0] = new float[Layers[0]];

            moveBiases = new float[Biases.Length][];
            moveWeights = new float[Weights.Length][][];


            for (int l = 1; l < Layers.Length; l++)
            {
                Neurons[l] = new float[Layers[l]];
                Z[l] = new float[Layers[l]];
                Biases[l] = new float[Layers[l]];
                MovingAverageBiases[l] = new float[Layers[l]];
                Weights[l] = new float[Layers[l]][];
                MovingAverage[l] = new float[Layers[l]][];

                error[l] = new float[Neurons[l].Length];
                moveBiases[l] = new float[Biases[l].Length];
                moveWeights[l] = new float[Weights[l].Length][];

                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    Biases[l][n] = GaussianRandom(0, 0.5f);


                    MovingAverageBiases[l][n] = 1;
                    Weights[l][n] = new float[Layers[l - 1]];
                    MovingAverage[l][n] = new float[Layers[l - 1]];
                    moveWeights[l][n] = new float[Weights[l][n].Length];


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

            for (int i = 0; i < Neurons[0].Length; i++)
                Neurons[0][i] = input[i];

            for(int l = 1; l < Layers.Length; l++)
            {
                //Parallel.For(0, Neurons[l].Length, (n) =>
                for (int n = 0; n < Neurons[l].Length; n++)
                {

                    Z[l][n] = 0;

                    //Parallel.For(0, Neurons[l - 1].Length, (prevN) =>
                    for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)

                    {
                        Z[l][n] += Weights[l][n][prevN] * Neurons[l - 1][prevN];
                    }//);

                    Z[l][n] += Biases[l][n];


                    if (l != Layers.Length - 1)
                        Neurons[l][n] = ActivationHidden(Z[l][n]);
                    else
                        Neurons[l][n] = ActivationOut(Z[l][n]);
                }//);
            }

            float[] output = new float[Neurons[Layers.Length - 1].Length];
            for (int i = 0; i < Neurons[Layers.Length - 1].Length; i++)
                output[i] = Neurons[Layers.Length - 1][i];

            return output;
        }


        public void Train(float[][] inputs, float[][] targets)
        {
            float totalCost = 0;
            for (int p = 0; p < inputs.Length; p++)
            {
                float[] input = inputs[p];
                float[] target = targets[p];
                float[] output = FeedForward(input);

                #region cost and plotting
                float cost = 0;
                for (int i = 0; i < output.Length; i++)
                    cost += (float)Math.Pow(output[i] - target[i], 2);

                totalCost += cost;

                /*int good = 0;
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

                if (cost == 0)
                    continue;

                //Computing the error
                //The error is basically the derivative of the cost by the z of that neuron at that place
                for (int i = 0; i < Layers[Layers.Length - 1]; i++)
                    error[Neurons.Length - 1][i] = 2 * (target[i] - output[i]) * ActivationOutDer(Z[Layers.Length - 1][i]);

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
                        moveBiases[l][n] += error[l][n];

                        for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)
                            moveWeights[l][n][prevN] += error[l][n] * Neurons[l - 1][prevN];
                    }
                }
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


        public float[][] TrainWithError(float[][] inputs, float[][] errors)
        {
            float totalCost = 0;
            float[][] returnErr = new float[inputs.Length][];

            for (int p = 0; p < inputs.Length; p++)
            {
                float[] input = inputs[p];

                #region cost and plotting
                float cost = 0;
                for (int i = 0; i < errors[p].Length; i++)
                    cost += Math.Abs(errors[p][i]);

                totalCost += cost;

                /*int good = 0;
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

                if (cost == 0)
                {

                    //continue;
                }

                //Computing last layer error (multiplying the arg by derivative of Z)
                //The error is basically the derivative of the cost by the z of that neuron at that place
                for (int i = 0; i < Layers[Layers.Length - 1]; i++)
                    error[Neurons.Length - 1][i] = errors[p][i] * ActivationOutDer(Z[Layers.Length - 1][i]);


                for (int l = Layers.Length - 1; l >= 2; l--)
                {
                    error[l - 1] = new float[Neurons[l - 1].Length];
                    for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)
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
                        moveBiases[l][n] += error[l][n];

                        for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)
                            moveWeights[l][n][prevN] += error[l][n] * Neurons[l - 1][prevN];
                    }
                }

                returnErr[p] = new float[Layers[0]];

                for (int prevN = 0; prevN < Neurons[0].Length; prevN++)
                {
                    for (int n = 0; n < Neurons[1].Length; n++)
                        returnErr[p][prevN] += error[1][n] * Weights[1][n][prevN];
                }


                /*for (int i = 0; i < Layers[0]; i++)
                    returnErr[p][i] = error[0][i] ;*/
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

            return returnErr;
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
            if (x >= 0)
                return x;
            return (float)Math.Exp(x) - 1;
        }

        private static float eLUPrime(float x)
        {
            if (x > 0)
                return 1;
            return (float)Math.Exp(x);
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
            {
                Directory.CreateDirectory(outputDir);
                outputDir += "\\";
            }
            else
                throw new Exception("Parent Dir does not exist");

            string jsonW = JsonSerializer.Serialize(this.Weights);
            string jsonB = JsonSerializer.Serialize(this.Biases);
            string jsonM = JsonSerializer.Serialize(this.MovingAverage);
            string jsonMB = JsonSerializer.Serialize(this.MovingAverageBiases);
            File.WriteAllText(outputDir + "weights", jsonW);
            File.WriteAllText(outputDir + "biases", jsonB);
            File.WriteAllText(outputDir + "movingAverage", jsonM);
            File.WriteAllText(outputDir + "movingAverageBiases", jsonMB);
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
            MovingAverage = JsonSerializer.Deserialize<float[][][]>(File.ReadAllText(inputDir + "movingAverage"));
            MovingAverageBiases = JsonSerializer.Deserialize<float[][]>(File.ReadAllText(inputDir + "movingAverageBiases"));
        }

        public NN2 Copy()
        {

            float[][][] w = new float[Layers.Length][][];
            float[][] b = new float[Layers.Length][];

            for (int l = 1; l < Layers.Length; l++)
            {
                b[l] = Biases[l];
                w[l] = new float[Layers[l]][];

                for (int n = 0; n < Layers[l]; n++)
                {
                    w[l][n] = new float[Layers[l - 1]];
                    for (int prevN = 0; prevN < Layers[l - 1]; prevN++)
                        w[l][n][prevN] = Weights[l][n][prevN];
                }
            }

            NN2 neural = new NN2(Layers, LearningRate);
            neural.Weights = w;
            neural.Biases = b;
            return neural;
        }

        #endregion
    }
}
