using System;
using System.Linq;
using Fiourp;
using ImageRecognition;

namespace CarDeepQ;

//Helped heavily by https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/cartpole_dqn.ipynb
public class DeepQAgent
{
    public float learningRate = 0.5f;
    public float gamma = 0.995f;
    public const int stateSize = 11;
    public const int actionSize = 6;
    public int[] layers = new int[] { stateSize, 64, 64, actionSize };
    public int BatchSize = 256;
    public int totalEpisodes = 5000;
    
    public float epsilon = 1;
    public float epsilonMin = 0.01f;
    public float epsilonDecay = 0.00005f;
    public float decayStep = 0;

    public int targetRefreshRate = 10000;

    public int gateTimeStepThreshold = 300;
    public float baseReward = 0;
    public float deathReward = -1f;
    public float gateReward = 1;

    public bool learning = true;   

    public Tuple<float[], int, float, float[], bool>[] memory = new Tuple<float[], int, float, float[], bool>[25000];
    public int iMemory = 0;
    public bool filledMemory = false;
    
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
            decayStep = 0.00001f;
            //Network.Load("C:\\Users\\zddng\\Documents\\Monogame\\CarDeepQ\\saves2\\net");
            TargetNetwork = Network.Copy();
            //epsilonDecay = (float)Math.Pow(epsilonMin, (double)1 / totalEpisodes);
            //epsilonDecay = (float)1 / (totalEpisodes + 1);
        }
        
    }

    public void Remember(float[] state, int action, float reward, float[] nextState, bool done)
    {
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
        decayStep += 1f;

        epsilon = epsilonMin + (1 - epsilonMin) * (float)Math.Exp(-epsilonDecay * decayStep);
        //epsilon -= 0.001f;

        var r = Rand.NextDouble();
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
    
    //This is where we train the algorithm
    public void Replay()
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
                if(filledMemory)
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

        float[][] inputs = new float[miniBatch.Length][];
        float[][] targets = new float[miniBatch.Length][];
        for(int i = 0; i < miniBatch.Length; i++)
        {
            Tuple<float[], int, float, float[], bool> info = miniBatch[i];
            float[] state = info.Item1;
            int action = info.Item2;
            float reward = info.Item3;
            if (info.Item3 == 10)
                Debug.Log("10");
            float[] nextState = info.Item4;
            bool done = info.Item5;

            float target;
            if (done)
                target = reward;
            else{
                float[] output = Network.FeedForward(nextState);
                int argMax = 0;
                float max = output[0];
                for(int k = 1; k < output.Length; k++)
                    if(output[k] > max){
                        max = output[k];
                        argMax = k;
                    }

                target = reward + gamma * TargetNetwork.FeedForward(nextState)[argMax];
            }

            float[] targetF = Network.FeedForward(state);
            targetF[action] = target;

            inputs[i] = state;
            targets[i] = targetF;
        }

        Network.Train(inputs, targets);

        if (epsilon > epsilonMin)
            epsilon *= epsilonDecay;
        //epsilonDecay *= 1.001f;
    }

    public void RefreshTargetNetwork()
        => TargetNetwork = Network.Copy();
}