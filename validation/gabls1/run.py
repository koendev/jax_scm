from scm.config import load_namelist
from scm.examples.gabls1 import get_gabls1
from scm.io.local import out_to_ds
from scm.mynn.model import init_model
from scm.time_stepping import simulate

if __name__ == "__main__":
    cfg = load_namelist("namelist_cn.yaml")
    sim = get_gabls1(Nz=64, plot=False)
    model = init_model(sim, implicit=True)
    out = simulate(model=model, sim=sim, cfg=cfg)

    # Save output
    ds = out_to_ds(out=out, sim=sim, time=out.t_s / 60 / 60)
    ds.to_netcdf(f"out_{sim.grid.Nz}.nc")
    print("Written to disk.")
