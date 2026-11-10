from acoustools.HighLevel import TopBottomContext


with TopBottomContext() as ctx:
    ctx.create_focus(ctx.origin)
    ctx.add_lev_sig()
    ctx.solution = ctx.solution * 0.5
    ctx.send_solution((60, 999))
    input()