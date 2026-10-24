"""Default: live GUI. --output: display-free PNG render."""
import argparse
import math
from pathlib import Path
from .core import make_system
from .data import STARS, PLANETS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--star', choices=STARS, default='sun')
    parser.add_argument('--planet', choices=PLANETS, action='append')
    parser.add_argument('--output', type=Path, help='Save a still image instead of opening the GUI')
    parser.add_argument('--years', type=float, default=0., help='Evolve before rendering (nonnegative)')
    args = parser.parse_args(argv)
    planets = args.planet or ['earth','jupiter']
    import matplotlib
    if args.output:
        matplotlib.use('Agg')
    from .plotting import App, draw
    import matplotlib.pyplot as plt
    try:
        if not math.isfinite(args.years) or args.years < 0:
            raise ValueError('--years must be nonnegative')
        if args.output:
            system = make_system(args.star,planets)
            system.step(round(args.years/system.dt))
            fig,ax = plt.subplots(figsize=(9,9))
            draw(ax,system,args.star,planets)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(args.output,dpi=150)
            plt.close(fig)
        else:
            app = App(args.star,planets)
            plt.show()
            return 0
    except ValueError as exc:
        parser.error(str(exc))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
