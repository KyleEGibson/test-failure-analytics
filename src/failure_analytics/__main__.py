"""Command line entry point.

    python -m failure_analytics generate              # synthetic EOS-100 production database
    python -m failure_analytics report --out site     # static HTML report
    python -m failure_analytics analyze --db results.db  # findings for any station results.db
"""

import argparse

from . import eos, report
from .eos_generator import generate


def main() -> None:
    parser = argparse.ArgumentParser(prog="failure_analytics")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="create the synthetic EOS-100 production database")
    gen.add_argument("--out", default=str(report.DEFAULT_EOS_DB))
    gen.add_argument("--seed", type=int, default=2026)
    rep = sub.add_parser("report", help="build the static HTML report")
    rep.add_argument("--out", default="site")
    rep.add_argument("--db", default=None, help="EOS database to analyze (default: regenerate)")
    ana = sub.add_parser("analyze", help="print root-cause findings for an EOS results database")
    ana.add_argument("--db", default=str(report.DEFAULT_EOS_DB))
    args = parser.parse_args()

    if args.command == "generate":
        print(generate(args.out, seed=args.seed))
    elif args.command == "report":
        print(report.build(args.out, args.db))
    else:
        for f in eos.analyze(eos.EOSDatabase(args.db)):
            where = f"{f.factor}={f.level}" if f.factor else "no assignable cause"
            print(f"{f.failure_mode:<28} {f.failures:>4}  {f.classification:<34} {where:<22} "
                  f"onset={f.onset_week}  spc={f.spc_signal_week}  retest-pass={f.no_fault_found_pct:.0f}%")


if __name__ == "__main__":
    main()
