using Fiourp;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Xml;
using static Fiourp.Input;
using static System.Collections.Specialized.BitVector32;

namespace CarDeepQ;

public class Car : Actor
{
    public float Rotation = 0; // In radians
    public Vector2 FrontVector;

    private const float friction = 0.1f;
    private const float accelSpeed = 0.66f;
    private const float turnForce = 0.05f;

    public Vector2 respawnPoint;
    public float respawnRot;

    private new BoxColliderRotated Collider;
    public RewardGate nextGate;

    public float TotalReward = 0;
    
    public Car(Vector2 position, float rotation) : base(position, 20, 10, 0, new Sprite(DataManager.Textures["car"]))
    {
        respawnPoint = Pos;
        Rotation = rotation;
        respawnRot = rotation;
        FrontVector = VectorHelper.Rotate(Vector2.UnitX, Rotation).Normalized();
        RemoveComponent(base.Collider);
        Collider = new BoxColliderRotated(Vector2.Zero, Width, Height, rotation, HalfSize);
        base.Collider = Collider;
        Collider.DebugDraw = false;
        AddComponent(Collider);
        Collider.Rotation = rotation;

        Sprite.Scale = Vector2.One * 0.06f;
        Sprite.Offset = new Vector2(9, 4);
        Sprite.Origin = new Vector2(DataManager.Textures["car"].Width, DataManager.Textures["car"].Height) / 2;
    }

    public override void Update()
    {
        base.Update();

        FrontVector = VectorHelper.Rotate(Vector2.UnitX, Rotation).Normalized();

        float factor = 1;
        bool done = false;
        Velocity -= Velocity * friction * factor;

        if (Input.GetKey(Keys.Right))
            Collider.Rotate(turnForce * factor, 0.002f, new List<Entity>(Engine.CurrentMap.Data.Solids), () => done = true);
        
        if (Input.GetKey(Keys.Left))
            Collider.Rotate(-turnForce * factor, 0.002f, new List<Entity>(Engine.CurrentMap.Data.Solids), () => done = true);

        Rotation = Collider.Rotation;

        if (Input.GetKey(Keys.Up))
            Velocity += FrontVector * accelSpeed * factor;
        if (Input.GetKey(Keys.Down))
            Velocity -= FrontVector * accelSpeed * factor;

        MoveX(Velocity.X, () => done = true);
        MoveY(Velocity.Y, () => done = true);
    }

    public bool Update(int action)
    {
        base.Update();
        
        FrontVector = VectorHelper.Rotate(Vector2.UnitX, Rotation).Normalized();

        float factor = 1;
        bool done = false;
        Velocity -= Velocity * friction * factor;

        if (action == 1)
            Collider.Rotate(turnForce * factor, 0.002f, new List<Entity>(Engine.CurrentMap.Data.Solids), () => done = true);

        if (action == 2)
            Collider.Rotate(-turnForce * factor, 0.002f, new List<Entity>(Engine.CurrentMap.Data.Solids), () => done = true);

        Rotation = Collider.Rotation;

        if (action == 0)
            Velocity += FrontVector * accelSpeed * factor;

        /*if(action == 4)
            Velocity -= FrontVector * accelSpeed * factor;*/
        
        MoveX(Velocity.X, () => done = true);
        MoveY(Velocity.Y, () => done = true);

        return done;
    }

    public override void Render()
    {
        Sprite.Rotation = Rotation + (float)Math.PI / 2;
        
        base.Render();
        //Drawing.Draw(Drawing.PointTexture, new Rectangle(Bounds.X + (int)HalfSize.X, Bounds.Y + (int)HalfSize.Y, Bounds.Width, Bounds.Height), Sprite.Color, Rotation,  Vector2.One / 2, Vector2.One);
    }

    public float[] GetState()
    {
        float[] state = new float[15];
        List<Ray> rays = new();
        for (int i = 0; i < 360; i += 45)
            rays.Add(new Ray(MiddlePos, VectorHelper.RotateDeg(FrontVector, i), 150)); //Distance to walls
        
        Ray.ShootRays(rays.ToArray()).CopyTo(state, 0);
        float vLen = VectorHelper.Projection(Velocity, FrontVector).Length() / (accelSpeed * 1 / friction);
        int sameDir = Math.Sign(Vector2.Dot(FrontVector, Velocity));

        if(sameDir == 1)
            state[8] = vLen; //Forward Speed
        else
            state[9] = vLen; //Backwards speed

        //Debug.LogUpdate("vlen  " + VectorHelper.Round(VectorHelper.Projection(Velocity, FrontVector)));
        //state[9] = Environment.Normalize(((BoxColliderRotated)nextGate.Collider).Rotation, 0, (float)Math.PI * 2);

        

        int sameDirX = Math.Sign(Vector2.Dot(VectorHelper.Normal(FrontVector), Velocity));
        float vLenX = VectorHelper.Projection(Velocity, VectorHelper.Normal(FrontVector)).Length() / 2.5f;

        if (sameDirX == 1)
            state[10] = vLenX; //Right speed
        else
            state[11] = vLenX; //Left speed

        Vector2 gateMidPos = ((BoxColliderRotated)nextGate.Collider).Center + nextGate.Pos;
        float dirGate = Environment.Normalize(VectorHelper.GetAngle(FrontVector, gateMidPos - MiddleExactPos), -3.14f, 3.14f);
        if (dirGate > 0.5f) dirGate = -dirGate + 1;

        state[12] = Environment.Normalize(Math.Min(Vector2.Distance(MiddlePos, gateMidPos), 200), 0, 200); //distance to gate
        state[13] = Environment.Normalize(((BoxColliderRotated)nextGate.Collider).Rotation, -3.14f, 3.14f); //Rotation of gate

        state[14] = Environment.Normalize(dirGate, 0, 0.5f); //Direction to gate

        /*Debug.Event(() =>
        {
            Drawing.DrawDottedLine(MiddlePos, gateMidPos, Color.Red, 1, 6, 4);
            Drawing.DrawLine(gateMidPos, gateMidPos + dirGate * 20, Color.Red, 1);
        });*/

        for (int i = 0; i < state.Length; i++)
            if (float.IsNaN(state[i]) || state[i] < 0 || state[i] > 1.1f)
                throw new Exception("States are not well normalized");

        return state;
    }

    public void Reset()
    {
        Pos = respawnPoint;
        Velocity = Vector2.Zero;

        Rotation = respawnRot;

        TotalReward = 0;
    }

    public class Ray
    {
        public Vector2 Begin;
        public Vector2 Direction;
        public float MaxLength;

        public float Distance;
        
        
        public Vector2 EndPoint;
        public bool Hit;

        public Ray(Vector2 begin, Vector2 direction, float maxLength)
        {
            Begin = begin;
            Direction = direction.Normalized();
            MaxLength = maxLength;
            EndPoint = begin + Direction * MaxLength;
        }

        public static float[] ShootRays(params Ray[] rays)
        {
            var walls = Engine.CurrentMap.Data.Solids;
            List<float> distances = new();
            foreach (Ray ray in rays)
            {
                foreach (Solid wall in walls)
                {
                    var rect = ((BoxColliderRotated)wall.Collider).Rect;
                    for (int i = 0; i <= 3; i++)
                    {
                        int next = i + 1;
                        if (next > 3)
                            next = 0;

                        if (Collision.LineIntersection(rect[i], rect[next], ray.Begin, ray.EndPoint) is Vector2 point)
                        {
                            ray.EndPoint = point;
                            ray.Hit = true;
                        }
                    }
                }

                ray.Distance = Vector2.Distance(ray.Begin, ray.EndPoint);
                /*Debug.Line(ray.Begin, ray.EndPoint, Color.Red, 1);
                Debug.PointUpdate(ray.EndPoint);*/
                float v = Ease.Reverse(Environment.Normalize(ray.Distance, 0, ray.MaxLength));
                distances.Add(Math.Clamp(v, 0, 1));
            }

            return distances.ToArray();
        }
    }
}