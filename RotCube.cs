using System.Runtime.InteropServices;
using Fiourp;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;

namespace CarDeepQ;

public class RotCube : Entity
{
    private bool mouseFollow = false;
    public RotCube(Vector2 position, int width, int height, bool mouseFollow) : base(position, width, height, new Sprite(Color.White))
    {
        this.mouseFollow = mouseFollow;
        if (mouseFollow)
        {
            RemoveComponent(base.Collider);
            Collider = new BoxColliderRotated(Vector2.Zero, Width, Height, 0, HalfSize);
            AddComponent(Collider);
            
            RemoveComponent(Sprite);
        }
    }

    public override void Update()
    {
        base.Update();

        if (mouseFollow && Input.GetMouseButton(MouseButton.Left))
            MiddlePos = Input.MousePos;

        if (Input.GetKey(Keys.A) && mouseFollow)
            ((BoxColliderRotated)Collider).Rotation += 0.01f;
        if (Input.GetKey(Keys.E) && mouseFollow)
            ((BoxColliderRotated)Collider).Rotation -= 0.01f;

        Sprite.Color = Color.White;
        foreach (RotCube cube in Engine.CurrentMap.Data.EntitiesByType[typeof(RotCube)])
        {
            if(cube != this && Collider.Collide(cube))
                Sprite.Color = Color.Blue;
        }
        
        if(Collider.Collide(Input.MousePos))
            Sprite.Color = Color.Blue;

        var r = new Car.Ray(MiddlePos, -Vector2.UnitY, 100000);
        /*Car.Ray.ShootRays(r);
        if(r.Hit)
            Debug.PointUpdate(Color.Aqua, r.EndPoint);*/
            //Debug.LogUpdate(r.EndPoint, Vector2.Distance(r.EndPoint, MiddlePos));
        /*else
            Debug.LogUpdate("No hit");*/
        
        var w = Engine.CurrentMap.Data.Solids[10];
        w.Sprite.Color = Color.Aqua;
        var rect = ((BoxColliderRotated)w.Collider).Rect;
        
        if(Collision.LineIntersection(Vector2.Zero, Vector2.UnitX* 10000, r.Begin, r.EndPoint) is Vector2 v)
            Debug.PointUpdate(v);
    }

    public override void Render()
    {
        base.Render();
        if(mouseFollow)
            Drawing.Draw(Drawing.PointTexture, new Rectangle(Bounds.X + (int)HalfSize.X, Bounds.Y + (int)HalfSize.Y, Bounds.Width, Bounds.Height), Sprite.Color, ((BoxColliderRotated)Collider).Rotation,  Vector2.One / 2, Vector2.One);
    }
}