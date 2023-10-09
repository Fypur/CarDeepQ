using Fiourp;
using Microsoft.Xna.Framework;

namespace CarDeepQ;

public class Wall : Solid
{
    public const float offsetX = -150;
    public const float offsetY = -150;
    public const float scale = 1;
    public Wall(Vector2 begin, Vector2 end) : base(begin, (int)(begin - end).Length(), 3, new Sprite(Color.White))
    {
        RemoveComponent(base.Collider);
        Collider = new BoxColliderRotated(Vector2.Zero, Width, 10, (end - begin).ToAngleDegrees(), Vector2.Zero);
        AddComponent(Collider);
    }
    
    public Wall(float beginX, float beginY, float endX, float endY) : this(new Vector2(beginX * Wall.scale + Wall.offsetX, beginY * Wall.scale + Wall.offsetY), new Vector2(endX * Wall.scale + Wall.offsetX, endY * Wall.scale + Wall.offsetY))
    { }

    public override void Render()
    {
        Drawing.Draw(Drawing.PointTexture, Bounds, Sprite.Color, ((BoxColliderRotated)Collider).Rotation, Vector2.Zero, Vector2.One);
    }
}