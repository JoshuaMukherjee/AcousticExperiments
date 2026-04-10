from acoustools.HighLevel import BottomWithReflectorContext

with BottomWithReflectorContext() as ctx:
    ctx.create_focus()
    ctx.visualise()
    