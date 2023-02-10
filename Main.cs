using System;
using System.Threading;
using Fiourp;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System.IO;
using ILGPU.Runtime;

namespace CarDeepQ;

public class Main : Game
{
    public static Main instance;
    private GraphicsDeviceManager _graphics;
    private SpriteBatch _spriteBatch;

    public static int episode;
    private Entity MainGate;
    static RewardGate[] RewardGates;
    static int gateIndex = 0;
    
    private System.IO.StreamWriter writer;
    Car car;
    
    public Main()
    {
        _graphics = new GraphicsDeviceManager(this);
        Content.RootDirectory = "Content";
        IsMouseVisible = true;
        instance = this;

       _graphics.SynchronizeWithVerticalRetrace = false;
        IsFixedTimeStep = false;

        /*var n5 = new NN5(new int[] { 10, 256, 256, 10 }, 0.01f);
        n5.Train(null, new float[10, 2] { { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 } });*/

        /*var n2 = new NN2(new int[] { 10, 256, 256, 10 }, 10);
        var n = new NN4(new int[] { 10, 256, 256, 10 }, 10);

        n2.Load("C:\\Users\\Administrateur\\Documents\\Monogame\\CarDeepQ\\net32");
        n.Load("C:\\Users\\Administrateur\\Documents\\Monogame\\CarDeepQ\\net32");

        //var o2 = n.FeedForward(new float[10] { 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 });

        float[][] input = new float[1][];
        float[][] target = new float[input.Length][];
        for(int i = 0; i < input.Length; i++)
        {
            input[i] = new float[10] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
            target[i] = new float[10] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        }*/

        /*n2.Train(input, target); //CPU
        n.Train(input, target); //GPU*/

        /*var o1 = n.FeedForward(input[0]);
        var o2 = n2.FeedForward(input[0]);*/

        //for (int i = 0; i < 1000; i++)
            //n2.Train(input, target); //CPU
            //n2.FeedForward(input[0]);


       //for (int i = 0; i < 1000; i++)
            //n.Train(input, target); //GPU
           // n.FeedForward(input[0]);
    }

    protected override void Initialize()
    {
        // TODO: Add your initialization logic here

        Engine.Initialize(_graphics, Content, 1280, 720, new RenderTarget2D(GraphicsDevice, 1600, 900), "");
        Debug.DebugMode = true;

        base.Initialize();

        /*for (int i = 0; i < 10000000; i++)
            Update(new GameTime(new TimeSpan(0),new TimeSpan(0),true));*/
    }

    protected override void LoadContent()
    {
        _spriteBatch = new SpriteBatch(GraphicsDevice);
        Drawing.Init(_spriteBatch, Content.Load<SpriteFont>("font"));

        Engine.CurrentMap = new Map(Vector2.Zero);
        Engine.Cam = new Camera(Vector2.Zero, 0, 1);
        
        DeepQAgent agent = new DeepQAgent();
        Engine.CurrentMap.Instantiate(new Environment(agent, true));
        InstantiateEnvironment();

        /*for(int i = 0; i < 3; i++)
            Engine.CurrentMap.Instantiate(new Environment(agent, false));*/

        //car = (Car)Engine.CurrentMap.Instantiate(new Car(new Vector2(258 * Wall.scale + Wall.offsetX, (1000 - 288) * Wall.scale + Wall.offsetY), (float)(3 * Math.PI / 2)));

        /*RewardGates = Environment.InstantiateGates();
        foreach(RewardGate gate in Environment.InstantiateGates())
            Engine.CurrentMap.Instantiate(gate);

        writer = new("/home/f/Documents/CarDeepQ/stuff/Respawn.txt", true);
        writer.AutoFlush = true;*/
        //Engine.CurrentMap.Instantiate(new RotCube(Vector2.Zero, 50, 50, true));
        //Engine.CurrentMap.Instantiate(new RotCube(Engine.ScreenSize / 2, 50, 50, false));
    }


    protected override void Update(GameTime gameTime)
    {
        if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed ||
            Keyboard.GetState().IsKeyDown(Keys.Escape))
            Exit();
        
        Input.UpdateState();

        if (Input.GetKeyDown(Keys.D1))
            Debug.DebugMode = !Debug.DebugMode;


        /*RewardGates[gateIndex].Update();
        
        if (RewardGates[gateIndex].Triggered)
        {
            gateIndex++;
            //Car.respawnPoint = Car.Pos;
            //Car.respawnRot = Car.Rotation;
        }
        
        if (Input.GetKeyDown(Keys.D3)){
            writer.WriteLine($"new Vector2({car.Pos.X}, {car.Pos.Y}), {car.Rotation}f, {gateIndex}f");
        }*/

        
        Engine.Update(gameTime);
        /*if(Engine.Deltatime != 0)
            Console.WriteLine(1 / Engine.Deltatime);*/

        Input.UpdateOldState();

        base.Update(gameTime);
    }

    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(Color.Black);
        
        GraphicsDevice.SetRenderTarget(Engine.RenderTarget);
        
        _spriteBatch.Begin();
         
        Engine.CurrentMap.Render();
        Engine.CurrentMap.UIRender();
        Engine.CurrentMap.UIOverlayRender();

        if(Debug.DebugMode)
        {
            Drawing.DebugString();
            Drawing.DebugPoint(10, 1);
            Drawing.DebugEvents();   
        }
        else
        {
            Debug.Clear();
        }
        
        _spriteBatch.End();
        
        GraphicsDevice.SetRenderTarget(null);
        
        _spriteBatch.Begin();
        
        Drawing.Draw(Engine.RenderTarget, new Rectangle(0, 0, (int)Engine.ScreenSize.X, (int)Engine.ScreenSize.Y));
        
        _spriteBatch.End();

        base.Draw(gameTime);
    }
    
    private void InstantiateEnvironment()
    {
        #region Walls
        
        Engine.CurrentMap.Instantiate(new Wall(240, 809, 200, 583));
        Engine.CurrentMap.Instantiate(new Wall(200, 583, 218, 395));
        Engine.CurrentMap.Instantiate(new Wall(218, 395, 303, 255));
        Engine.CurrentMap.Instantiate(new Wall(303, 255, 548, 173));
        Engine.CurrentMap.Instantiate(new Wall(548, 173, 764, 179));
        Engine.CurrentMap.Instantiate(new Wall(764, 179, 1058, 198));
        Engine.CurrentMap.Instantiate(new Wall(1055, 199, 1180, 215));
        Engine.CurrentMap.Instantiate(new Wall(1177, 215, 1220, 272));
        Engine.CurrentMap.Instantiate(new Wall(1222, 273, 1218, 367));
        Engine.CurrentMap.Instantiate(new Wall(1218, 367, 1150, 437));
        Engine.CurrentMap.Instantiate(new Wall(1150, 437, 1044, 460));
        Engine.CurrentMap.Instantiate(new Wall(1044, 460, 757, 600));
        Engine.CurrentMap.Instantiate(new Wall(757, 600, 1099, 570));
        Engine.CurrentMap.Instantiate(new Wall(1100, 570, 1187, 508));
        Engine.CurrentMap.Instantiate(new Wall(1187, 507, 1288, 443));
        Engine.CurrentMap.Instantiate(new Wall(1288, 443, 1463, 415));
        Engine.CurrentMap.Instantiate(new Wall(1463, 415, 1615, 478));
        Engine.CurrentMap.Instantiate(new Wall(1617, 479, 1727, 679));
        Engine.CurrentMap.Instantiate(new Wall(1727, 679, 1697, 874));
        Engine.CurrentMap.Instantiate(new Wall(1694, 872, 1520, 964));
        Engine.CurrentMap.Instantiate(new Wall(1520, 964, 1100, 970));
        Engine.CurrentMap.Instantiate(new Wall(1105, 970, 335, 960));
        Engine.CurrentMap.Instantiate(new Wall(339, 960, 264, 899));
        Engine.CurrentMap.Instantiate(new Wall(263, 897, 238, 803));
        Engine.CurrentMap.Instantiate(new Wall(317, 782, 274, 570));
        Engine.CurrentMap.Instantiate(new Wall(275, 569, 284, 407));
        Engine.CurrentMap.Instantiate(new Wall(284, 407, 363, 317));
        Engine.CurrentMap.Instantiate(new Wall(363, 317, 562, 240));
        Engine.CurrentMap.Instantiate(new Wall(562, 240, 1114, 284));
        Engine.CurrentMap.Instantiate(new Wall(1114, 284, 1120, 323));
        Engine.CurrentMap.Instantiate(new Wall(1120, 323, 1045, 377));
        Engine.CurrentMap.Instantiate(new Wall(1045, 378, 682, 548));
        Engine.CurrentMap.Instantiate(new Wall(682, 548, 604, 610));
        Engine.CurrentMap.Instantiate(new Wall(604, 612, 603, 695));
        Engine.CurrentMap.Instantiate(new Wall(605, 695, 702, 713));
        Engine.CurrentMap.Instantiate(new Wall(703, 712, 1128, 642));
        Engine.CurrentMap.Instantiate(new Wall(1129, 642, 1320, 512));
        Engine.CurrentMap.Instantiate(new Wall(1323, 512, 1464, 497));
        Engine.CurrentMap.Instantiate(new Wall(1464, 497, 1579, 535));
        Engine.CurrentMap.Instantiate(new Wall(1579, 535, 1660, 701));
        Engine.CurrentMap.Instantiate(new Wall(1660, 697, 1634, 818));
        Engine.CurrentMap.Instantiate(new Wall(1634, 818, 1499, 889));
        Engine.CurrentMap.Instantiate(new Wall(1499, 889, 395, 883));
        Engine.CurrentMap.Instantiate(new Wall(395, 883, 330, 838));
        Engine.CurrentMap.Instantiate(new Wall(330, 838, 315, 782));
        Engine.CurrentMap.Instantiate(new Wall(319, 798, 306, 725));
        Engine.CurrentMap.Instantiate(new Wall(276, 580, 277, 543));
        Engine.CurrentMap.Instantiate(new Wall(603, 639, 622, 590));
        Engine.CurrentMap.Instantiate(new Wall(599, 655, 621, 704));
        Engine.CurrentMap.Instantiate(new Wall(1074, 571, 1115, 558));
        Engine.CurrentMap.Instantiate(new Wall(1314, 516, 1333, 511));
        Engine.CurrentMap.Instantiate(new Wall(1692, 875, 1706, 830));
        Engine.CurrentMap.Instantiate(new Wall(277, 912, 255, 872));
        Engine.CurrentMap.Instantiate(new Wall(1214, 262, 1225, 288));
        Engine.CurrentMap.Instantiate(new Wall(1601, 470, 1625, 490));
        Engine.CurrentMap.Instantiate(new Wall(1119, 644, 1139, 634));
        Engine.CurrentMap.Instantiate(new Wall(687, 710, 719, 710));
        Engine.CurrentMap.Instantiate(new Wall(1721, 664, 1727, 696));
        Engine.CurrentMap.Instantiate(new Wall(1015, 392, 1065, 362));
        Engine.CurrentMap.Instantiate(new Wall(1091, 572, 1104, 568));
        Engine.CurrentMap.Instantiate(new Wall(1157, 528, 1233, 478));
        
        #endregion
    }
}
