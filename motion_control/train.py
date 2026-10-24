"""Train policy and value networks for the motion control problem.

    python -m motion_control.train --dims 2 4 6 8
"""

from pathlib import Path

from neural_hjb.cli import train_kwargs, training_parser
from neural_hjb.experiment import train_grid

from .problem import MotionControl

MODEL_DIR = Path(__file__).resolve().parent / "models"


def main() -> None:
    parser = training_parser("Train the motion control HJB solver", rollout_steps=20)
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 4, 6, 8], help="State dimensions")
    args = parser.parse_args()
    if any(n <= 0 for n in args.dims):
        raise SystemExit("--dims must be positive")
    problems = [MotionControl(state_dim=n) for n in args.dims]
    train_grid(problems, args.variants, args.seeds, MODEL_DIR, jobs=args.jobs, force=args.force,
               **train_kwargs(args))


if __name__ == "__main__":
    main()
