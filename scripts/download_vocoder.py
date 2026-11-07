#!/usr/bin/env python3
"""Download the universal HiFi-GAN generator used by ming024/FastSpeech2.

The zip is about 50 MB and is gitignored. Both languages use this vocoder
because neither corpus is LJSpeech.
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

PINNED = "d4e79eb52e8b01d24703b2dfc0385544092958f3"
URL = (
    "https://github.com/ming024/FastSpeech2/raw/{}/hifigan/generator_universal.pth.tar.zip".format(PINNED)
)


def parse_args():
    parser = argparse.ArgumentParser(description="Download ming024 universal HiFi-GAN weights.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from config_resolve import FS2_ROOT

    destination = FS2_ROOT / "hifigan" / "generator_universal.pth.tar"
    if destination.is_file() and not args.force:
        print("Vocoder already present at {}".format(destination))
        return
    print("Downloading {}".format(URL))
    with urlopen(URL) as response:
        payload = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            chosen = None
            for name in members:
                if name.endswith(".pth.tar") or name.endswith(".pth"):
                    chosen = name
                    break
            if chosen is None:
                raise SystemExit("Zip did not contain a checkpoint. Members: {}".format(members))
            with archive.open(chosen) as source, open(destination, "wb") as target:
                target.write(source.read())
    else:
        destination.write_bytes(payload)
    print("Wrote {} ({} bytes)".format(destination, destination.stat().st_size))


if __name__ == "__main__":
    main()
