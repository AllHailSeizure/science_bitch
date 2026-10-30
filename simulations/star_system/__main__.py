"""Default: live GUI. --output: display-free PNG render."""
import argparse
import math
from pathlib import Path
from .core import make_system, SOLAR_RADIUS_AU
from .data import STARS, PLANETS
from .populations import Population, with_population


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--star', choices=STARS, default='sun')
    parser.add_argument('--planet', choices=PLANETS, action='append')
    parser.add_argument('--output', type=Path, help='Save a still image instead of opening the GUI')
    parser.add_argument('--years', type=float, default=0., help='Evolve before rendering (nonnegative)')
    parser.add_argument('--population', choices=('ring', 'disk', 'belt'))
    parser.add_argument('--count', type=int, default=200, help='Massless particles (1-2000)')
    parser.add_argument('--inner', type=float, default=2., help='Inner population radius in AU from star')
    parser.add_argument('--outer', type=float, help='Outer radius in AU; defaults to inner for rings, 3.5 otherwise')
    parser.add_argument('--seed', type=int, default=0, help='Nonnegative population seed')
    parser.add_argument('--angles', choices=('random', 'even'), default='random')
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
        population = None
        if args.population:
            outer = args.outer if args.outer is not None else args.inner if args.population == 'ring' else 3.5
            population = Population(args.population, args.count, args.inner, outer, args.seed, args.angles)
            # Validate before opening a window, as well as on each rebuild.
            with_population(make_system(args.star, planets), population,
                            STARS[args.star].radius*SOLAR_RADIUS_AU)
        if args.output:
            system = make_system(args.star,planets)
            if population is not None:
                system = with_population(system, population, STARS[args.star].radius*SOLAR_RADIUS_AU)
            system.step(round(args.years/system.dt))
            fig,ax = plt.subplots(figsize=(9,9))
            draw(ax,system,args.star,planets)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(args.output,dpi=150)
            plt.close(fig)
        else:
            app = App(args.star,planets, population=population)
            plt.show()
            return 0
    except ValueError as exc:
        parser.error(str(exc))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
