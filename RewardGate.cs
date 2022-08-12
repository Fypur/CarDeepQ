using System;
using System.Collections.Generic;
using Fiourp;
using Microsoft.Xna.Framework;

namespace CarDeepQ;

public class RewardGate : Trigger
{
    public bool Triggered = false;
    public RewardGate(Vector2 begin, Vector2 end)
        : base(begin, (int)(begin - end).Length(), 10, new() { typeof(Car) }, new Sprite(Color.White))
    {
        RemoveComponent(base.Collider);
        Collider = new BoxColliderRotated(Vector2.Zero, Width, 10, (end - begin).ToAngle(), Vector2.Zero);
        AddComponent(Collider);
        Active = false;
    }

    public RewardGate(float beginX, float beginY, float endX, float endY) : this(new Vector2(beginX * Wall.scale + Wall.offsetX, (1000 - beginY) * Wall.scale + Wall.offsetY), new Vector2(endX * Wall.scale + Wall.offsetX, (1000 - endY) * Wall.scale + Wall.offsetY)) { }

    public override void Awake()
    {
        base.Awake();
        
        Sprite.Color = Color.Green;
    }

    public override void Render()
    {
        Drawing.Draw(Drawing.pointTexture, Bounds, Sprite.Color, ((BoxColliderRotated)Collider).Rotation, Vector2.Zero, Vector2.One);
    }

    public override void Update()
    {
        Triggered = false;
        base.Update();
    }

    public override void OnTriggerEnter(Entity entity)
    {
        base.OnTriggerEnter(entity);

        Triggered = true;
    }
}