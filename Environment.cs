using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Xml.Schema;
using Fiourp;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using ScottPlot;

namespace CarDeepQ;

public class Environment : Entity
{
    public Car Car;
    private DeepQAgent agent;
    
    public int timeStep = 0;
    private int targetTimeStep = 0;
    private int gateTimeStep = 0;
    private static bool respawn1 = true;

    public RewardGate[] RewardGates = InstantiateGates();
    public Tuple<Vector2, float, int>[] RespawnPoints = LoadRespawnPoints();
    private int gateIndex = 0;
    public static Plot p;
    public static List<double> pointsX = new();
    public static List<double> pointsY = new();

    public Environment(DeepQAgent agent, bool rendered) : base(Vector2.Zero)
    {
        this.agent = agent;
        Car = new Car(new Vector2(230 * Wall.scale + Wall.offsetX, (1000 - 400) * Wall.scale + Wall.offsetY), (float)(3 * Math.PI / 2));
        Car.nextGate = RewardGates[0];
        Engine.CurrentMap.Instantiate(Car);
        Car.Active = false;
        Car.Visible = false;
        
        Visible = rendered;

        p = new();
        p.Title("Reward by episodes");
    }

    public override void Update()
    {
        base.Update();

        float[] state = Car.GetState();
        int action = agent.Act(state);
        bool done = Car.Update(action);

        done = done || gateTimeStep > agent.gateTimeStepThreshold || timeStep > 10000;

        float reward = agent.baseReward;
        RewardGates[gateIndex].Update();
        if (RewardGates[gateIndex].Triggered)
        {
            reward = agent.gateReward;
            RewardGates[gateIndex].Triggered = false;
            gateIndex++;
            if(gateIndex >= RewardGates.Length)
                gateIndex = 0;
            Car.nextGate = RewardGates[gateIndex];
            gateTimeStep = 0;

            //Car.respawnPoint = Car.Pos;
            //Car.respawnRot = Car.Rotation;
        }
        
        float[] nextState = Car.GetState();
        if (done)
            reward = agent.deathReward;

        Car.TotalReward += reward;
        Debug.LogUpdate($"Episode {Main.episode}, Score {Car.TotalReward}, Epsilon: {agent.epsilon}, Timestep: {timeStep}");

        if(Input.GetKeyDown(Keys.D2)){
            agent.decayStep = 10000;
        }

        if(Input.GetKeyDown(Keys.D9) && agent.epsilonMin < 0.99f){
            agent.epsilonMin += 0.01f;
        }

        if(Input.GetKeyDown(Keys.D8) && agent.epsilonMin > 0.01f){
            agent.epsilonMin -= 0.01f;
        }

        if(agent.learning)
            agent.Remember(state, action, reward, nextState, done);

        if (done)
        {
            if(Car.TotalReward >= 100)
                Console.ForegroundColor = ConsoleColor.Yellow;
            else if(Car.TotalReward >= 50)
                Console.ForegroundColor = ConsoleColor.Green;
            else if(Car.TotalReward >= 30)
                Console.ForegroundColor = ConsoleColor.Blue;
            else
                Console.ForegroundColor = ConsoleColor.Red;

            Console.WriteLine($"Episode {Main.episode}, Score {Car.TotalReward}, Epsilon: {agent.epsilon}");
            Console.ForegroundColor = ConsoleColor.Gray;

            Car.nextGate = RewardGates[gateIndex];

            pointsX.Add(Main.episode);
            pointsY.Add(Car.TotalReward);
            if (Main.episode % 20 == 0)
            {
                p.AddSignalXY(pointsX.ToArray(), pointsY.ToArray());
                p.SaveFig("final.png");
                p.Clear();
            }

            Main.episode++;
            timeStep = 0;
            gateTimeStep = 0;


            if(agent.learning && (agent.filledMemory || agent.iMemory > agent.BatchSize))
                agent.Replay();

            var r = RespawnPoints[Rand.NextInt(0, RespawnPoints.Length)];
            Car.respawnPoint = r.Item1;
            Car.respawnRot = r.Item2; // + Rand.NextFloat(-0.7f, 0.7f);

            Car.Reset();
            
            gateIndex = r.Item3;

            if (Main.episode >= 1000 && Main.episode % 1000 == 0)
            {
                agent.Network.Save("C:\\Users\\zddng\\Documents\\Monogame\\CarDeepQ\\netAutoSave\\");
            }
            /*if(Main.episode > 100 && Car.TotalReward == agent.deathReward)
                agent.epsilon += 0.03f;*/
        }

        /*if (Input.GetKeyDown(Keys.S) && Visible)
            agent.Network.Save("/home/f/Documents/CarDeepQ/saves/net3");
        if (Input.GetKeyDown(Keys.L) && Visible)
            agent.Network.Load("/home/f/Documents/CarDeepQ/saves/net3");*/

        if(Input.GetKeyDown(Keys.S))
            agent.Network.Save("C:\\Users\\zddng\\Documents\\Monogame\\CarDeepQ\\netManualSave\\");
        
        
        
        
        timeStep++;
        gateTimeStep++;

        if (Visible)
        {
            targetTimeStep++;
            if (targetTimeStep >= agent.targetRefreshRate)
            {
                agent.RefreshTargetNetwork();
                targetTimeStep = 0;
            }
        }
    }

    public override void Render()
    {
        base.Render();

        Car.Render();
        RewardGates[gateIndex].Render();
    }

    public static float Normalize(float value, float min, float max)
        => (value - min) / (max - min);

    public static Tuple<Vector2, float, int>[] LoadRespawnPoints()
    {
        if(!respawn1)
            return new Tuple<Vector2, float, int>[]
            {
                new(new Vector2(230 * Wall.scale + Wall.offsetX, (1000 - 400) * Wall.scale + Wall.offsetY), (float)(3 * Math.PI / 2), 0),
                new(new Vector2(102, 239), 5.3524036f, 2),
                new(new Vector2(257, 93), 5.802414f, 4),
                new(new Vector2(589, 53), 0.25923973f, 7),
                new(new Vector2(946, 103), 0.34923965f, 10),
                new(new Vector2(618, 396), 2.8492374f, 14),
                new(new Vector2(786, 466), 0.16924016f, 16),
                new(new Vector2(1130, 337), 5.8724165f, 20),
                new(new Vector2(1382, 337), 0.4592405f, 22),
                new(new Vector2(1518, 552), 1.1392394f, 25),
                new(new Vector2(1393, 739), 2.5692382f, 27),
                new(new Vector2(1155, 762), 3.1892376f, 29),
                new(new Vector2(383, 768), 3.2592375f, 33),
            };
        else
            return new Tuple<Vector2, float, int>[]
            {
                new(new Vector2(230 * Wall.scale + Wall.offsetX, (1000 - 400) * Wall.scale + Wall.offsetY), (float)(3 * Math.PI / 2), 0),
            };

    }

    public static RewardGate[] InstantiateGates()
    {
        return new[]
        {
            new RewardGate(187, 435, 311, 451),
            new RewardGate(307, 537, 171, 555),
            new RewardGate(234, 681, 345, 628),
            new RewardGate(408, 682, 363, 788),
            new RewardGate(428, 816, 481, 712),
            new RewardGate(568, 733, 543, 854),
            new RewardGate(678, 858, 675, 710),
            new RewardGate(852, 708, 855, 848),
            new RewardGate(995, 836, 985, 705),
            new RewardGate(1059, 710, 1076, 821),
            new RewardGate(1078, 667, 1172, 572),
            new RewardGate(997, 616, 1076, 532),
            new RewardGate(967, 492, 909, 566),
            new RewardGate(788, 512, 839, 438),
            new RewardGate(790, 405, 781, 285),
            new RewardGate(891, 302, 899, 427),
            new RewardGate(1004, 434, 1027, 334),
            new RewardGate(1139, 344, 1084, 452),
            new RewardGate(1171, 502, 1233, 416),
            new RewardGate(1305, 454, 1243, 556),
            new RewardGate(1365, 588, 1408, 480),
            new RewardGate(1487, 472, 1524, 587),
            new RewardGate(1642, 508, 1575, 432),
            new RewardGate(1608, 360, 1709, 419),
            new RewardGate(1744, 324, 1625, 296),
            new RewardGate(1609, 231, 1727, 190),
            new RewardGate(1617, 66, 1541, 163),
            new RewardGate(1487, 135, 1510, 14),
            new RewardGate(1344, 16, 1328, 150),
            new RewardGate(1077, 142, 1067, 14),
            new RewardGate(909, 16, 900, 130),
            new RewardGate(718, 138, 698, 20),
            new RewardGate(551, 18, 567, 132),
            new RewardGate(445, 138, 413, 13),
            new RewardGate(379, 154, 243, 80),
            new RewardGate(357, 221, 203, 182)
        };
    }
}