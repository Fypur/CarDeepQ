using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Xml;
using Fiourp;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using static Fiourp.Input;

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
    
    public Car(Vector2 position, float rotation) : base(position, 20, 10, 0, new Sprite(Color.Red))
    {
        respawnPoint = Pos;
        Rotation = rotation;
        respawnRot = rotation;
        FrontVector = VectorHelper.Rotate(Vector2.UnitX, Rotation).Normalized();
        RemoveComponent(base.Collider);
        Collider = new BoxColliderRotated(Vector2.Zero, Width, Height, rotation, HalfSize);
        base.Collider = Collider;
        AddComponent(Collider);
    }

    public override void Update()
    {
        base.Update();

        FrontVector = VectorHelper.Rotate(Vector2.UnitX, Rotation).Normalized();

        Velocity -= Velocity * friction;

        if (Input.GetKey(Keys.Z))
            Velocity += FrontVector * accelSpeed;

        if (Input.GetKey(Keys.D))
            Rotation += turnForce;
        
        if (Input.GetKey(Keys.Q))
            Rotation -= turnForce;
        
        if(Input.GetKey(Keys.S))
            Velocity -= FrontVector * accelSpeed;

        if (Input.GetKey(Keys.V))
            Pos = Input.MousePos;

        while (Rotation > Math.PI * 2)
            Rotation -= (float)Math.PI * 2;
        while (Rotation < 0)
            Rotation += (float)Math.PI * 2;
        
        Collider.Rotation = Rotation;
        
        Sprite.Color = Color.Red;
        
        MoveX(Velocity.X, Reset);
        MoveY(Velocity.Y, Reset);

        /*float[] state = GetState();
        for (int i = 0; i < state.Length; i++)
            Debug.LogUpdate(state[i]);*/
    }

    public bool Update(int action)
    {
        base.Update();
        
        FrontVector = VectorHelper.Rotate(Vector2.UnitX, Rotation).Normalized();

        int factor = 1;
        Velocity -= Velocity * friction * factor;

        if (action == 1 || action == 3)
            Rotation += turnForce * factor;
        
        if (action == 2 || action == 4)
            Rotation -= turnForce * factor;

        if (action == 0 || action == 3 || action == 4)
            Velocity += FrontVector * accelSpeed * factor;

        if(action == 5)
            Velocity -= FrontVector * accelSpeed * factor;
        
        while (Rotation > Math.PI * 2)
            Rotation -= (float)Math.PI * 2;
        while (Rotation < 0)
            Rotation += (float)Math.PI * 2;
        
        Collider.Rotation = Rotation;
        
        Sprite.Color = Color.Red;

        bool done = false;
        MoveX(Velocity.X, () => done = true);
        MoveY(Velocity.Y, () => done = true);

        return done;
    }

    public override void Render()
    {
        //Drawing.Draw(Bounds, Rotation, Color.Red);
        Drawing.Draw(Drawing.PointTexture, new Rectangle(Bounds.X + (int)HalfSize.X, Bounds.Y + (int)HalfSize.Y, Bounds.Width, Bounds.Height), Sprite.Color, Rotation,  Vector2.One / 2, Vector2.One);
    }

    public float[] GetState()
    {
        float[] state = new float[11];
        List<Ray> rays = new();
        for (int i = 0; i < 360; i += 45)
            rays.Add(new Ray(MiddlePos, VectorHelper.RotateDeg(FrontVector, i), 150));
        
        Ray.ShootRays(rays.ToArray()).CopyTo(state, 0);
        float vLen = VectorHelper.Projection(Velocity, FrontVector).Length() / (accelSpeed * 1 / friction);
        int sameDir = Math.Sign(Vector2.Dot(FrontVector, Velocity));

        if(sameDir == 1)
            state[8] = vLen;
        else
            state[9] = vLen;

        //Debug.LogUpdate("vlen  " + VectorHelper.Round(VectorHelper.Projection(Velocity, FrontVector)));
        //state[9] = Environment.Normalize(((BoxColliderRotated)nextGate.Collider).Rotation, 0, (float)Math.PI * 2);

        state[10] = Environment.Normalize(Math.Min(Vector2.Distance(MiddlePos, nextGate.MiddlePos), 200), 0, 200);

        for(int i = 0; i < state.Length; i++)
            if(float.IsNaN(state[i]) || state[i] < -0.1f || state[i] > 1.1f)
                {}

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
                distances.Add(Ease.Reverse(Environment.Normalize(ray.Distance, 0, ray.MaxLength)));
            }

            return distances.ToArray();
        }
    }
}