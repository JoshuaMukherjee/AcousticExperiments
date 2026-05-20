from acoustools.HighLevel import TopWithReflectorContext

with TopWithReflectorContext() as ctx:
    ctx.create_focus()
    ctx.visualise(size=0.06)
    