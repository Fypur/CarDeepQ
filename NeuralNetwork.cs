using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Fiourp;

namespace CarDeepQ
{
    //Made with help of https://towardsdatascience.com/building-a-neural-network-framework-in-c-16ef56ce1fef
    public class NeuralNetwork
    {
        public float LearningRate = 0.01f;
        public readonly int[] layers;

        private float[][] neurons;
        private float[][] z;
        private float[][][] weights;
        private float[][] biases;

        public NeuralNetwork(int[] layers, float learningRate) : base()
        {
            this.layers = layers;
            LearningRate = learningRate;

            InitNeurons();
            InitBiases();
            InitWeights();

            /*var n1 = Sigmoid(neurons[0][0] * weights[0][0][0] + biases[1][0]);
            var n2 = Sigmoid(n1 * weights[1][0][0] + biases[2][0]);
            Console.WriteLine(FeedForward(new float[] { neurons[0][0] })[0]);*/
        }

        public NeuralNetwork(int[] layers, float learningRate, float[][][] weights, float[][] biases)
        {
            this.layers = new int[layers.Length];
            for(int i = 0; i < layers.Length; i++)
                this.layers[i] = layers[i];

            LearningRate = learningRate;

            InitNeurons();


            /*weights.CopyTo(this.weights, 0);
            biases.CopyTo(this.biases, 0);

            {}

            
            {}*/

            this.weights = new float[weights.Length][][];
            this.biases = new float[biases.Length][];

            for(int i = 0; i < weights.Length; i++){
                this.weights[i] = new float[weights[i].Length][];
                for(int j = 0; j < weights[i].Length; j++){
                    this.weights[i][j] = new float[weights[i][j].Length];
                    for(int k = 0; k < weights[i][j].Length; k++)
                        this.weights[i][j][k] = weights[i][j][k];
                }
            }
                

            for(int i = 0; i < biases.Length; i++){
                this.biases[i] = new float[biases[i].Length];
                for(int j = 0; j < biases[i].Length; j++)
                    this.biases[i][j] = biases[i][j];
            }
        }

        //Create empty Array to store the neurons in the network
        void InitNeurons()
        {
            List<float[]> neuronsList = new();

            foreach (int layerNumNeurons in layers)
                neuronsList.Add(new float[layerNumNeurons]);

            neurons = neuronsList.ToArray();
            z = neuronsList.ToArray();
        }

        void InitBiases()
        {
            List<float[]> biasesList = new();
            biasesList.Add(new float[layers[0]]);
            for (int i = 1; i < layers.Length; i++)
            {
                biasesList.Add(new float[layers[i]]);
                for (int j = 0; j < biasesList[i].Length; j++)
                    biasesList[i][j] = GaussianRandom(0, 0.5f);
            }

            biases = biasesList.ToArray();
        }

        void InitWeights()
        {
            List<float[][]> weightsList = new();

            weightsList.Add(new float[0][]);
            //Foreach layer
            for (int i = 1; i < layers.Length; i++)
            {
                List<float[]> layersWeightList = new();
                float std = (float)Math.Sqrt(2.0 / layers[i - 1]);

                //Foreach neuron in layer
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    List<float> neuronWeightsList = new();

                    //Foreach neuron in the next layer
                    for (int k = 0; k < layers[i - 1]; k++)
                        neuronWeightsList.Add(GaussianRandom(0, std)); //He Initialisation for ReLue
                    //neuronWeightsList.Add(GaussianRandom(0, 0.5f));

                    layersWeightList.Add(neuronWeightsList.ToArray());
                }

                weightsList.Add(layersWeightList.ToArray());
            }

            weights = weightsList.ToArray();
        }

        public float[] FeedForward(float[] inputs)
        {
            //Set Input Neurons
            for (int i = 0; i < inputs.Length; i++)
                neurons[0][i] = inputs[i];

            //Console.WriteLine($"layer 1, neuron 0, has a value of {neurons[0][0]}");

            //Foreach layer except input layer
            for (int i = 1; i < layers.Length; i++)
            {
                //Foreach neuron
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    float value = 0;
                    //Foreach neuron in the previous layer
                    for (int k = 0; k < neurons[i - 1].Length; k++)
                    {
                        float previousNeuron = neurons[i - 1][k];
                        float weight = weights[i][j][k];
                        if(float.IsNaN(weight) || weight > 1000)
                        {}
                        value += previousNeuron * weight;
                    }

                    if(float.IsNaN(biases[i][j]))
                    {}

                    z[i][j] = value + biases[i][j];
                    if (i == layers.Length - 1)
                        neurons[i][j] = z[i][j];
                    else
                        neurons[i][j] = ReLU(z[i][j]);
                    //Console.WriteLine($"layer {i}, neuron {j} has a value of {neurons[i][j]}");
                }
            }

            float[] result = new float[neurons[neurons.Length - 1].Length];
            neurons[neurons.Length - 1].CopyTo(result, 0);
            return result;
        }

        #region Train Simple

        public void Train(float[] inputs, float[] target)
        {
            float cost = 0;
            float[] output = FeedForward(inputs);

            string s = "result: ";
            for (int i = 0; i < output.Length; i++)
                s += Math.Round(output[i], 2) + " ";
            //Console.WriteLine(s);

            for (int i = 0; i < output.Length; i++)
            {
                cost += (float)Math.Pow(output[i] - target[i], 2);
            }


            float[][] error = new float[layers.Length][];
            for (int i = 0; i < layers.Length; i++)
                error[i] = new float[layers[i]];

            for (int j = 0; j < neurons[layers.Length - 1].Length; j++)
                error[layers.Length - 1][j] = 2 * (output[j] - target[j]) * 1; //SigmoidPrime(z[layers.Length - 1][j]);

            //Computing the error for every single neuron
            for (int l = layers.Length - 2; l >= 0; l--)
            {
                for (int k = 0; k < neurons[l].Length; k++)
                {
                    //Not sure about this part
                    float e = 0;
                    for (int j = 0; j < neurons[l + 1].Length; j++)
                        e += error[l + 1][j] * weights[l + 1][j][k];

                    error[l][k] = e * ReLUPrime(z[l][k]);
                }
            }

            //error[0] = new float[0];

            //Moving biases
            for (int l = 1; l < layers.Length; l++)
            for (int k = 0; k < neurons[l].Length; k++)
            {
                biases[l][k] -= error[l][k] * LearningRate;
                if (float.IsNaN(biases[l][k]) || Math.Abs(biases[l][k]) > 1000000)
                {
                }
            }


            //Moving weights
            for (int l = 1; l < layers.Length; l++)
            {
                for (int j = 0; j < neurons[l].Length; j++)
                {
                    for (int k = 0; k < neurons[l - 1].Length; k++)
                    {
                        float removed = neurons[l - 1][k] * error[l][j];
                        weights[l][j][k] -= neurons[l - 1][k] * error[l][j] * LearningRate;
                        if (float.IsNaN(weights[l][j][k]) || Math.Abs(weights[l][j][k]) > 1000000)
                        {
                        }
                    }
                }
            }

            cost /= 2;

            /*if (cost < 0.1f)
                Console.ForegroundColor = ConsoleColor.Green;
            else
                Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("cost: " + cost);
            Console.ForegroundColor = ConsoleColor.Gray;*/
        }
        
        #endregion
        
        public void Train(float[][] inputs, float[][] targets)
        {
            float[][][] moveWeights = new float[weights.Length][][];
            for (int i = 0; i < weights.Length; i++)
            {
                moveWeights[i] = new float[weights[i].Length][];
                for (int j = 0; j < weights[i].Length; j++)
                    moveWeights[i][j] = new float[weights[i][j].Length];
            }

            float[][] moveBiases = new float[biases.Length][];
            for (int i = 0; i < biases.Length; i++)
                moveBiases[i] = new float[biases[i].Length];

            float cost = 0;
            for (int t = 0; t < inputs.Length; t++)
            {
                var input = inputs[t];
                var target = targets[t];
                float costForExample = 0;

                float[] output = FeedForward(input);
                
                #region Visualisation

                /*string s = "result: ";
                for (int i = 0; i < output.Length; i++)
                    s += Math.Round(output[i], 2) + " ";
                //Console.WriteLine(s);

                for (int i = 0; i < output.Length; i++)
                    costForExample += (float)Math.Pow(output[i] - target[i], 2);
                costForExample /= 2;

                if (costForExample < 0.1f)
                    Console.ForegroundColor = ConsoleColor.Green;
                else
                    Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("costForExample: " + costForExample);
                Console.ForegroundColor = ConsoleColor.Gray;*/

                #endregion
                
                float[][] error = new float[layers.Length][];
                for (int i = 0; i < layers.Length; i++)
                    error[i] = new float[layers[i]];

                for (int j = 0; j < neurons[layers.Length - 1].Length; j++){
                    error[layers.Length - 1][j] = 2 * (output[j] - target[j]) * 1; //SigmoidPrime(z[layers.Length - 1][j]);
                    if(float.IsNaN(error[layers.Length - 1][j]) || error[layers.Length - 1][j] > 100)
                    {}
                }
                    

                //Computing the error for every single neuron
                for (int l = layers.Length - 2; l >= 0; l--)
                {
                    for (int k = 0; k < neurons[l].Length; k++)
                    {
                        float e = 0;
                        for (int j = 0; j < neurons[l + 1].Length; j++)
                            e += error[l + 1][j] * weights[l + 1][j][k];

                        error[l][k] = e * ReLUPrime(z[l][k]);
                    }
                }

                //error[0] = new float[0];

                //Moving biases
                for (int l = 1; l < layers.Length; l++)
                for (int k = 0; k < neurons[l].Length; k++)
                {
                    if(float.IsNaN(error[l][k]) || error[l][k] > 1000)
                    {}
                    moveBiases[l][k] -= error[l][k] * LearningRate;
                }


                //Moving weights
                for (int l = 1; l < layers.Length; l++)
                {
                    for (int j = 0; j < neurons[l].Length; j++)
                    {
                        for (int k = 0; k < neurons[l - 1].Length; k++)
                        {
                            float removed = neurons[l - 1][k] * error[l][j];
                            if(float.IsNaN(removed))
                            {}
                            moveWeights[l][j][k] -= neurons[l - 1][k] * error[l][j] * LearningRate;
                        }
                    }
                }

            }

            for (int i = 0; i < weights.Length; i++)
            for (int j = 0; j < weights[i].Length; j++)
            for (int k = 0; k < weights[i][j].Length; k++)
                weights[i][j][k] += moveWeights[i][j][k] / inputs.Length;

            for (int i = 0; i < biases.Length; i++)
            for (int j = 0; j < biases[i].Length; j++)
                biases[i][j] += moveBiases[i][j] / inputs.Length;

            //cost /= inputs.Length;

            /*if (cost < 0.1f)
                Console.ForegroundColor = ConsoleColor.Green;
            else
                Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("cost: " + cost);
            Console.ForegroundColor = ConsoleColor.Gray;*/
        }

        public static float Sigmoid(float x)
            => (float)(1 / (1 + Math.Exp(-x)));

        private float SigmoidPrime(float x)
            => Sigmoid(x) * (1 - Sigmoid(x));

        private float ReLU(float x)
        {
            if (x >= 0)
                return x;
            return 0;
        }

        private float ReLUPrime(float x)
        {
            if (x > 0)
                return 1;
            return 0;
        }

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
            string jsonW = JsonSerializer.Serialize(this.weights);
            string jsonB = JsonSerializer.Serialize(this.biases);
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

            weights = JsonSerializer.Deserialize<float[][][]>(jsonW);
            biases = JsonSerializer.Deserialize<float[][]>(jsonB);
        }

        public NeuralNetwork Copy()
        {
            NeuralNetwork n = new NeuralNetwork(layers, LearningRate, weights, biases);
            return n;
        }
    }
}
