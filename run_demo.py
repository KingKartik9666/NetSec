"""Run the offline prototype from any working directory."""

import sys


if __name__ == "__main__":
    try:
        from experiment.demo import main
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency {exc.name!r}. Activate the project environment, then run "
            "python -m pip install -r requirements.txt (see README.md).",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    raise SystemExit(main())
