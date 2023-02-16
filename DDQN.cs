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
    public int[] layers = new int[] { stateSize, 64, 64, 64, actionSize };
    public int BatchSize = 64;
    public int totalEpisodes = 50000;

    public float epsilon = 1;
    public float epsilonMin = 0.03f;
    public float epsilonDecay = 0.0001f;
    public float decayStep = 0;

    public int targetRefreshRate = 1000;

    public int gateTimeStepThreshold = 500;
    public float baseReward = -0.03f;
    public float deathReward = -5;
    public float gateReward = 10;

    public bool learning = true;

    public Tuple<float[], int, float, float[], bool>[] memory = new Tuple<float[], int, float, float[], bool>[100000];
    public int iMemory = 0;
    public bool filledMemory = false;
    private bool saveMemory = true;

    public NN2 BaseNetwork;
    public NN2 AdvantageNetwork;
    public NN2 ValueNetwork;

    public NN2 TargetNetwork;

    public DDQN()
    {
        BaseNetwork = new NN2(layers, learningRate);
        TargetNetwork = BaseNetwork.Copy();

        if (!learning)
        {
            epsilon = 0;
            epsilonDecay = 1;
        }
        else
        {
            //decayStep = 1000000;
            //Network.Load("C:\\Users\\zddng\\Documents\\Monogame\\CarDeepQ\\netManualSave\\");

            TargetNetwork = BaseNetwork.Copy();

            //epsilonDecay = (float)Math.Pow(epsilonMin, (double)1 / totalEpisodes);
            //epsilonDecay = (float)1 / (totalEpisodes + 1);
        }
        if (!saveMemory)
        {
            memory = System.Text.Json.JsonSerializer.Deserialize<Tuple<float[], int, float, float[], bool>[]>(System.IO.File.ReadAllText("C:\\Users\\zddng\\Documents\\Monogame\\CarDeepQ\\memory"));
            filledMemory = true;
        }
    }

    public float[] FeedForward(float[] input)
    {
        input = BaseNetwork.FeedForward(input);
        float[] advantage = AdvantageNetwork.FeedForward(input);
        float[] value = ValueNetwork.FeedForward(input);

        float[] QValues = new float[layers[layers.Length - 1]];

        float advantageAverage = 0;
        for (int i = 0; i < QValues.Length; i++)
            advantageAverage += advantage[i];

        advantageAverage /= advantage.Length;

        for (int i = 0; i < QValues.Length; i++)
            QValues[i] = value[i] + (advantage[i] - advantageAverage);

        return QValues;
    }


    public float[] FeedForward(float[] input, out float advantageAverage)
    {
        input = BaseNetwork.FeedForward(input);
        float[] advantage = AdvantageNetwork.FeedForward(input);
        float[] value = ValueNetwork.FeedForward(input);

        float[] QValues = new float[layers[layers.Length - 1]];

        advantageAverage = 0;
        for (int i = 0; i < QValues.Length; i++)
            advantageAverage += advantage[i];

        advantageAverage /= advantage.Length;

        for (int i = 0; i < QValues.Length; i++)
            QValues[i] = value[i] + (advantage[i] - advantageAverage);

        return QValues;
    }

    //This is where we train the algorithm
    public void Replay()
    {
        if (!filledMemory)
            return;

        Tuple<float[], int, float, float[], bool>[] miniBatch = Sample();





        float[][] inputs = new float[miniBatch.Length][];
        float[][] advTargets = new float[miniBatch.Length][];
        float[] vTargets = new float[miniBatch.Length][];


        for (int i = 0; i < miniBatch.Length; i++)
        {
            Tuple<float[], int, float, float[], bool> info = miniBatch[i];
            float[] state = info.Item1;
            int action = info.Item2;
            float reward = info.Item3;
            float[] nextState = info.Item4;
            bool done = info.Item5;


            if (done)
            {

            }
            else
            {
                float[] outputNxtState = FeedForward(nextState);

                float nextStateValue = outputNxtState.Average();


                vTargets[i] = reward + nextStateValue;


                advTargets[i] = new float[outputNxtState.Length];
                for(int k = 0; k < outputNxtState.Length; k++)
                    advTargets[i][k] = outputNxtState[i] - nextStateValue;

                advTargets[i][action] = 

                



            }

            //Get Advantage targets




            //Get Value Target


            //Backprop both

            //Get Error





            

            float target;
            if (done)
                target = reward; //if we are on a terminal state
            else
            {
                float[] output = BaseNetwork.FeedForward(nextState, out float advAntageAverage); //for on a non terminal state
                

                if (reward != -0.01f)
                { }

                target = reward + gamma * TargetNetwork.FeedForward(nextState)[argMax];







            }

            float[] targetF = BaseNetwork.FeedForward(state);
            targetF[action] = target;

            inputs[i] = state;
            targets[i] = targetF;
        }

        BaseNetwork.Train(inputs, targets);
    }

    public void Remember(float[] state, int action, float reward, float[] nextState, bool done)
    {
        if (filledMemory && saveMemory)
        {
            System.IO.File.WriteAllText("C:\\Users\\zddng\\Documents\\Monogame\\CarDeepQ\\memory", System.Text.Json.JsonSerializer.Serialize(memory));
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
        => TargetNetwork = BaseNetwork.Copy();
}