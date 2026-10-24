"""Train policy and value networks for the resource allocation problem.

    python -m resource_allocation.train
"""

from pathlib import Path

from neural_hjb.cli import train_kwargs, training_parser
from neural_hjb.experiment import train_grid

from .problem import ResourceAllocation

MODEL_DIR = Path(__file__).resolve().parent / "models"


def main() -> None:
    parser = training_parser("Train the resource allocation HJB solver", rollout_steps=40)
    parser.add_argument("--gamma", type=float, default=0.5, help="Growth rate of reinvested production")
    parser.add_argument("--horizon", type=float, default=4.0, help="Horizon T")
    args = parser.parse_args()
    problems = [ResourceAllocation(gamma=args.gamma, T=args.horizon)]
    train_grid(problems, args.variants, args.seeds, MODEL_DIR, jobs=args.jobs, force=args.force,
               **train_kwargs(args))


if __name__ == "__main__":
    main()
