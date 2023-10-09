using System;
using System.Linq;
using Fiourp;
using ILGPU;
using ILGPU.Runtime;

namespace CarDeepQ;

//Helped heavily by https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/cartpole_dqn.ipynb
public class DDQN
{
    public float learningRate = 0.01f;
    public float gamma = 0.95f;
    public const int stateSize = 15;
    public const int actionSize = 6;
    public const int inBetweenSize = 16;
    public int BatchSize = 64;

    public int[] layers = new int[] { stateSize, 64, 64, inBetweenSize };
    public int[] layersV = new int[] { inBetweenSize, 64, 1 };
    public int[] layersA = new int[] { inBetweenSize, 64, 64, actionSize };
    public int totalEpisodes = 50000;

    public float epsilon = 1;
    public float epsilonMin = 0.03f;
    public float epsilonDecay = 0.0004f;
    public float decayStep = 0;

    public int targetRefreshRate = 1000;

    public int gateTimeStepThreshold = 500;
    public float baseReward = -0.01f;
    public float deathReward = -10;
    public float gateReward = 10;

    public bool learning = true;

    public Tuple<float[], int, float, float[], bool>[] memory = new Tuple<float[], int, float, float[], bool>[20000];
    public int iMemory = 0;
    public bool filledMemory = false;
    private bool saveMemory = false;

    public NN2 BaseNetwork;
    public NN2 AdvantageNetwork;
    public NN2 ValueNetwork;

    public NN2 TargetBaseNetwork;
    public NN2 TargetAdvantageNetwork;
    public NN2 TargetValueNetwork;

    private float[][] inputs;
    private float[][] inputVA;
    private float[][] targets;
    private float[][] vTargets;
    private float[][] aTargets;

    public DDQN()
    {
        BaseNetwork = new NN2(layers, learningRate);
        AdvantageNetwork = new NN2(layersA, learningRate);
        ValueNetwork = new NN2(layersV, learningRate);


        if (!learning)
        {
            epsilon = 0;
            epsilonDecay = 1;
        }
        else
        {
            //decayStep = 1000000;
            //Network.Load("C:\\Users\\zddng\\Documents\\Monogame\\CarDeepQ\\netManualSave\\");

            RefreshTargetNetwork();

            //epsilonDecay = (float)Math.Pow(epsilonMin, (double)1 / totalEpisodes);
            //epsilonDecay = (float)1 / (totalEpisodes + 1);
        }
        if (!saveMemory)
        {
            memory = System.Text.Json.JsonSerializer.Deserialize<Tuple<float[], int, float, float[], bool>[]>(System.IO.File.ReadAllText("C:\\Users\\Administrateur\\Documents\\Monogame\\CarDeepQ\\memory"));
            filledMemory = true;
        }

        inputs = new float[BatchSize][];
        inputVA = new float[BatchSize][];
        targets = new float[BatchSize][];
        vTargets = new float[BatchSize][];
        aTargets = new float[BatchSize][];
    }

    public float[] FeedForward(float[] input)
    {
        input = BaseNetwork.FeedForward(input);
        float[] advantage = AdvantageNetwork.FeedForward(input);
        float[] value = ValueNetwork.FeedForward(input);

        float[] QValues = new float[actionSize];

        float advantageAverage = advantage.Average();

        for (int i = 0; i < QValues.Length; i++)
            QValues[i] = value[0] + (advantage[i] - advantageAverage);

        return QValues;
    }


    public float[] FeedForward(float[] input, out float advantageAverage)
    {
        input = BaseNetwork.FeedForward(input);
        float[] advantage = AdvantageNetwork.FeedForward(input);
        float[] value = ValueNetwork.FeedForward(input);

        float[] QValues = new float[actionSize];

        advantageAverage = advantage.Average();

        for (int i = 0; i < QValues.Length; i++)
            QValues[i] = value[0] + (advantage[i] - advantageAverage);

        return QValues;
    }

    public float[] TargetFeedForward(float[] input)
    {
        input = TargetBaseNetwork.FeedForward(input);
        float[] advantage = TargetAdvantageNetwork.FeedForward(input);
        float[] value = TargetValueNetwork.FeedForward(input);

        float[] QValues = new float[actionSize];

        float advantageAverage = advantage.Average();

        for (int i = 0; i < QValues.Length; i++)
            QValues[i] = value[0] + (advantage[i] - advantageAverage);

        return QValues;
    }

    //https://datascience.stackexchange.com/questions/54023/dueling-network-gradient-with-respect-to-advantage-stream
    public void Replay()
    {
        if (!filledMemory)
            return;

        Tuple<float[], int, float, float[], bool>[] miniBatch = Sample();

        for (int i = 0; i < miniBatch.Length; i++)
        {
            Tuple<float[], int, float, float[], bool> info = miniBatch[i];
            float[] state = info.Item1;
            int action = info.Item2;
            float reward = info.Item3;
            float[] nextState = info.Item4;
            bool done = info.Item5;

            float[] output = FeedForward(state);

            vTargets[i] = new float[1];

            float target;
            if (done)
                target = reward; //if we are on a terminal state
            else
            {
                float[] outputNxtState = FeedForward(nextState); //for on a non terminal state

                int argMax = ArgMax(outputNxtState);

                if (reward != -0.01f)
                { }

                //target = reward + gamma * TargetFeedForward(nextState)[argMax];
                target = reward + gamma * output[argMax];
                //Debug.LogUpdate("TARGET NETWORK DOES NOT WORK YET!!!!!! (I THINK)");
            }

            //output[action] = target;
            inputs[i] = state; //inputs for training Main Base Network

            inputVA[i] = new float[AdvantageNetwork.Neurons[0].Length];
            for (int k = 0; k < AdvantageNetwork.Neurons[0].Length; k++)
                inputVA[i][k] = AdvantageNetwork.Neurons[0][k]; //Inputs for training A and V networks

            targets[i] = output;

            vTargets[i][0] = 2 * (target - output[action]); //V Gradient according to link, (gradForA)
            //delta E over delta Q is equal to 0 except for one action so we don't need a for loop



            aTargets[i] = new float[output.Length];
            float scalarB = -(float)1/actionSize * vTargets[i][0]; //Look at link to know what this is
            for (int k = 0; k < output.Length; k++)
            {
                aTargets[i][k] = scalarB; //Advantage Gradient = scalarB + delta E / delta Q which is 0 except for one action

                if(k == action)
                    aTargets[i][k] += 2 * (target - output[action]);
            }
            
        }

        float[][] advError = AdvantageNetwork.TrainWithError(inputVA, aTargets);
        float[][] valueError = ValueNetwork.TrainWithError(inputVA, vTargets);

        float[][] baseError = new float[miniBatch.Length][];

        for (int i = 0; i < miniBatch.Length; i++)
        {
            baseError[i] = new float[inBetweenSize];
            for (int k = 0; k < inBetweenSize; k++)
                baseError[i][k] = advError[i][k] + valueError[i][k];
        }

        BaseNetwork.TrainWithError(inputs, baseError);
    }


    public void Remember(float[] state, int action, float reward, float[] nextState, bool done)
    {
        if (filledMemory && saveMemory)
        {
            System.IO.File.WriteAllText("C:\\Users\\Administrateur\\Documents\\Monogame\\CarDeepQ\\memory", System.Text.Json.JsonSerializer.Serialize(memory));
            saveMemory = false;
        }

        memory[iMemory] = new(state, action, reward, nextState, done);
        iMemory++;
        if (iMemory > memory.Length - 1)
        {
            iMemory = 0;
            filledMemory = true;
        }
    }

    public int Act(float[] state)
    {
        if (!filledMemory)
            return Rand.NextInt(0, actionSize);

        /*string j =System.Text.Json.JsonSerializer.Serialize(memory);
        System.IO.File.WriteAllText("C:\\Users\\zddng\\Documents\\Monogame\\CarDeepQ\\memory", j);*/

        decayStep += 1f;

        epsilon = epsilonMin + (1 - epsilonMin) * (float)Math.Exp(-epsilonDecay * decayStep);
        //epsilon -= 0.001f;
        var r = Rand.NextDouble();
        //r = 1;
        //r = 1;
        if (r < epsilon)
        {
            int r2 = Rand.NextInt(0, actionSize);
            return r2;
        }


        float[] netValues = FeedForward(state);
        float max = netValues[0];
        int argMax = 0;
        for (int i = 1; i < netValues.Length; i++)
        {
            if (netValues[i] > max)
            {
                max = netValues[i];
                argMax = i; int r2 = Rand.NextInt(0, actionSize);
            }
            else if (netValues[i] == max && Rand.NextDouble() > 0.5)
            {
                argMax = i;
            }
        }

        //Debug.LogUpdate(argMax);
        return argMax;
    }

    public Tuple<float[], int, float, float[], bool>[] Sample()
    {
        //Create MiniBatch

        int[] miniBatchIndexes = new int[BatchSize];
        for (int i = 0; i < BatchSize; i++)
            miniBatchIndexes[i] = -1;

        Tuple<float[], int, float, float[], bool>[] miniBatch = new Tuple<float[], int, float, float[], bool>[BatchSize];
        for (int i = 0; i < BatchSize; i++)
        {
            int r;

            void SetR()
            {
                if (filledMemory)
                    r = Rand.NextInt(0, memory.Length);
                else
                    r = Rand.NextInt(0, iMemory);
            }

            SetR();
            while (miniBatchIndexes.Contains(r))
                SetR();
            miniBatchIndexes[i] = r;
            miniBatch[i] = memory[r];
        }

        return miniBatch;
    }

    /*int r = Rand.NextInt(0, memory.Length);
        int limit =  r + BatchSize;
        int count = 0;
        for (int i = r; i < limit; i++)
        {
            if(i + 1 > memory.Length)
            {
                limit = BatchSize - count;
                i = 0;
            }

            miniBatch[count] = memory[r];
            count++;
        }*/

    public int ArgMax(float[] input)
    {
        int argMax = 0;
        float max = input[0];
        for (int k = 1; k < input.Length; k++)
            if (input[k] > max)
            {
                max = input[k];
                argMax = k;
            }

        return argMax;
    }

    public void RefreshTargetNetwork()
    {
        TargetBaseNetwork = BaseNetwork.Copy();
        TargetAdvantageNetwork = AdvantageNetwork.Copy();
        TargetValueNetwork = ValueNetwork.Copy();
    }
}