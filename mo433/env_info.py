"""Record the software and hardware of this machine in $MO433_DATA/env_<host>.json.

Run once per environment (laptop, Colab). The versions go to the report's appendix and the
GPU name confirms that every timed run used the same GPU model (report Sec. 5.3).
"""

import argparse

from mo433.io import banner, data_dir, environment, host_label, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="output path (default: env_<host>.json)")
    args = parser.parse_args()

    out = args.out or data_dir() / f"env_{host_label()}.json"
    env = environment()
    save_json(env, out)

    banner(f"Environment ({env['host']})")
    print(f"Python  {env['python']}  |  {env['platform']}")
    for name, version in env["packages"].items():
        print(f"  {name:<13} {version}")
    dev = env["device"]
    print(f"CPU     {dev['cpu']} ({dev['cpu_count']} threads)")
    print(f"GPU     {dev.get('gpu', 'none')}")
    print(f"git     {env['git']['commit']} (dirty={env['git']['dirty']})")
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
