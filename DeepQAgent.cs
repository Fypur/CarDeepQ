using System;
using System.Linq;
using Fiourp;
using ILGPU;
using ILGPU.Runtime;

namespace CarDeepQ;

//Helped heavily by https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/cartpole_dqn.ipynb
public class DeepQAgent
{
    public float learningRate = 0.001f;
    public float gamma = 0.99f;
    public const int stateSize = 15;
    public const int actionSize = 3;
    public int[] layers = new int[] { stateSize, 32, 32, 32, actionSize };
    public int totalEpisodes = 50000;
    
    public float epsilon = 1;
    public float epsilonMin = 0.03f;
    public float epsilonDecay = 0.00007f;
    public float decayStep = 0;
    public int BatchSize = 64;

    public int targetRefreshRate = 100;

    public int gateTimeStepThreshold = 500;
    public float baseReward = -0.1f;
    public float deathReward = -10;
    public float gateReward = 30;

    public bool learning = true;

    public Tuple<float[], int, float, float[], bool>[] memory = new Tuple<float[], int, float, float[], bool>[10000];
    public int iMemory = 0;
    public bool filledMemory = false;
    private bool saveMemory = false;
    
    public NN2 Network;
    public NN2 TargetNetwork;

    public DeepQAgent()
    {
        Network = new NN2(layers, learningRate);
        TargetNetwork = Network.Copy();

        if (!learning)
        {
            epsilon = 0;
            epsilonDecay = 1;
        }
        else
        {
            //decayStep = 1000000;
            //Network.Load("C:\\Users\\zddng\\Documents\\Monogame\\CarDeepQ\\netManualSave\\");

            TargetNetwork = Network.Copy();

            //epsilonDecay = (float)Math.Pow(epsilonMin, (double)1 / totalEpisodes);
            //epsilonDecay = (float)1 / (totalEpisodes + 1);
        }
        if (!saveMemory)
        {
            /*memory = System.Text.Json.JsonSerializer.Deserialize<Tuple<float[], int, float, float[], bool>[]>(System.IO.File.ReadAllText("C:\\Users\\Administrateur\\Documents\\Monogame\\CarDeepQ\\memory"));
            filledMemory = true;*/
        }
    }

    //This is where we train the algorithm
    public void Replay()
    {
        if (!filledMemory)
            return;

        Tuple<float[], int, float, float[], bool>[] miniBatch = Sample();

        float[][] inputs = new float[miniBatch.Length][];
        float[][] targets = new float[miniBatch.Length][];
        for (int i = 0; i < miniBatch.Length; i++)
        {
            Tuple<float[], int, float, float[], bool> info = miniBatch[i];
            float[] state = info.Item1;
            int action = info.Item2;
            float reward = info.Item3;
            float[] nextState = info.Item4;
            bool done = info.Item5;

            float target;
            if (done)
                target = reward; //if we are on a terminal state
            else
            {
                float[] output = Network.FeedForward(nextState); //for on a non terminal state
                int argMax = 0;
                float max = output[0];
                for (int k = 1; k < output.Length; k++)
                    if (output[k] > max)
                    {
                        max = output[k];
                        argMax = k;
                    }

                if (reward != -0.01f)
                { }

                target = reward + gamma * Network.FeedForward(nextState)[argMax];
            }

            float[] targetF = Network.FeedForward(state);
            targetF[action] = target;

            inputs[i] = state;
            targets[i] = targetF;
        }

        Network.Train(inputs, targets);
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
        float[] netValues = Network.FeedForward(state);
        float max = netValues[0];
        int argMax = 0;
        for(int i = 1; i < netValues.Length; i++)
        {
            if (netValues[i] > max)
            {
                max = netValues[i];
                argMax = i;int r2 = Rand.NextInt(0, actionSize);
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

    public void RefreshTargetNetwork()
        => TargetNetwork = Network.Copy();
}