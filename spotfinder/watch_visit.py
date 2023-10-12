from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
import shutil
import requests

R = "\033[31m"
G = "\033[32m"
B = "\033[34m"
P = "\033[35m"
GRAY = "\033[37m"
BOLD = "\033[1m"
NC = "\033[0m"

DCSERVER = "https://ssx-dcserver.diamond.ac.uk"

progress = itertools.cycle(["▸▹▹▹▹", "▹▸▹▹▹", "▹▹▸▹▹", "▹▹▹▸▹", "▹▹▹▹▸"])


@lru_cache
def _get_auth_headers() -> dict[str, str]:
    try:
        TOKEN = os.environ["DCSERVER_TOKEN"]
    except KeyError:
        sys.exit(f"{R}{BOLD}Error: No credentials file. Please set {_credentials}{NC}")

    return {"Authorization": "Bearer " + TOKEN}


class DCIDFetcher:
    highest_dcid: int | None
    visit: str

    def __init__(self, visit: str, since: int | None):
        self.visit = visit
        self.highest_dcid = since

    def fetch(self) -> list[dict]:
        params = {}
        if self.highest_dcid is not None:
            params = {"since_dcid": self.highest_dcid}

        resp = requests.get(
            f"{DCSERVER}/visit/{self.visit}/dc",
            headers=_get_auth_headers(),
            params=params,
        )
        if resp.status_code == 403:
            sys.exit(f"{R}{BOLD}Error: Unauthorised: " + resp.json()["detail"] + NC)
        resp.raise_for_status()
        dcs = resp.json()

        # If we got collections, update our latest DCID
        if dcs:
            self.highest_dcid = max(
                self.highest_dcid or 0, *[x["dataCollectionId"] for x in dcs]
            )
        return dcs


def run():
    parser = argparse.ArgumentParser(
        description="Watch a visit for data collections and launch spotfinding"
    )
    parser.add_argument("visit", help="The name of the visit to watch.")
    parser.add_argument(
        "command",
        help="Command and arguments to run upon finding a new data collection. Use '{}' to interpose the image name. Use '{nimages}' to interpose the image count, and '{startimagenumber}' for the starting image number.",
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "-t",
        help="Time (in seconds) to wait between requests. Default: %(default)ss",
        default=5,
        type=int,
        dest="wait",
    )
    parser.add_argument(
        "--all",
        help="Trigger on all collections, including existing.",
        action="store_true",
    )
    parser.add_argument(
        "--since", help="Ignore collections with DCIDs lower than this.", type=int
    )
    args = parser.parse_args()

    # Prepare the execute function
    def _prepare_command(image: Path | str, nimages: str | int, startImageNumber: str | int) -> list[str]:
        command = args.command[:]
        has_image = False
        for i in range(len(command)):
            if "{nimages}" in command[i]:
                command[i] = command[i].replace("{nimages}", str(nimages))
            if "{nimages}" in command[i]:
                command[i] = command[i].replace("{nimages}", str(nimages))
            if "{startimagenumber}" in command[i]:
                command[i] = command[i].replace("{startimagenumber}", str(startImageNumber))
            if "{}" in command[i]:
                command[i] = command[i].replace("{}", str(image))
                has_image = True

        if not has_image:
            command.append(str(image))

        return command

    if args.command:
        if not Path(args.command[0]).is_file() and not shutil.which(args.command[0]):
            sys.exit(f"Error: Command {args.command[0]} appears not to exist")

        print(
            f"Running command on data collection:{BOLD}{P}",
            shlex.join(_prepare_command("<filename>", "<nimages>", "<startimagenumber>")) + NC,
        )

    fetcher = DCIDFetcher(args.visit, since=args.since)

    if not args.all:
        # Grab all existing DCIDs first, so we don't get overloaded
        if existing := fetcher.fetch():
            print(f"Discarding {BOLD}{len(existing)}{NC} pre-existing data collections")
        else:
            print(f"No existing data collections.")

    print(f"Waiting for more data collections in {BOLD}{args.visit}{NC}...\n")
    while True:
        if new_dcs := sorted(fetcher.fetch(), key=lambda x: x["dataCollectionId"]):
            for dc in new_dcs:
                print(
                    f"\rFound new datacollection: {BOLD}{dc['dataCollectionId']}{NC} ({dc['startTime'].replace('T', ' ')})"
                )
                image_path = Path(dc["imageDirectory"]) / dc["fileTemplate"]
                print(
                    f"    {BOLD}{dc['numberOfImages']}{NC} images in {B}{image_path}{NC}"
                )
                if args.command:
                    start = time.monotonic()
                    _command = _prepare_command(
                        image=image_path, nimages=dc["numberOfImages"], startImageNumber=dc["startImageNumber"],
                    )
                    print("+", shlex.join(_command))
                    proc = subprocess.run(_command)
                    elapsed = time.monotonic() - start
                    if proc.returncode == 0:
                        print(f"Command done after {BOLD}{elapsed:.1f}{NC} s")
                    else:
                        print(
                            f"{R}{BOLD}Command ended with error in {BOLD}{elapsed:.1f}{NC}{R} s{NC}"
                        )
                print()

            print(f"Waiting for more data collections in {BOLD}{args.visit}{NC}...\n")

        print(f" {next(progress)}\r", end="")
        try:
            time.sleep(args.wait)
        except KeyboardInterrupt:
            # Closing while sleeping is perfectly normal
            print("        ")
            sys.exit()
