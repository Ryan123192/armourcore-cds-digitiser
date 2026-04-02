import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/run_single.py <input_path> [template_id]")
        return 1

    input_path = sys.argv[1]
    template_id = sys.argv[2] if len(sys.argv) > 2 else "cds_regular_500x600"

    cmd = [
        sys.executable,
        "-m",
        "armourcore_cds.cli",
        input_path,
        "--template",
        template_id,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
