#!/usr/bin/env python3
"""
Parse AlphaQubit output .npy data files.

Usage:
    python inspect_npy.py output/si1000_syndromes_z_20260123_140440.npy
    python inspect_npy.py output/si1000_syndromes_z_20260123_140440.npy 2>&1 | head -35
"""

import argparse
import os
import struct
import sys
import numpy as np


def analyze(path: str) -> None:
    # allow_pickle=True to detect non-array .npy files (e.g. dicts, objects)
    raw = np.load(path, allow_pickle=True)

    print(f"\n{'='*64}")
    print(f"  File         : {os.path.basename(path)}")
    print(f"  Python type  : {type(raw)}")

    # .npy can store arbitrary Python objects via pickle (e.g. dict, list)
    if not isinstance(raw, np.ndarray):
        print(f"  Content      : {raw}")
        return

    arr: np.ndarray = raw
    print(f"  ndarray type : confirmed (no extra metadata)")
    print(f"  Shape        : {arr.shape}")
    print(f"  Dtype        : {arr.dtype}")
    print(f"  Flags        :")
    print(f"    C-contiguous : {arr.flags['C_CONTIGUOUS']}")
    print(f"    Writeable    : {arr.flags['WRITEABLE']}")
    print(f"    OwnData      : {arr.flags['OWNDATA']}")

    # parse .npy binary header and verify file size
    with open(path, "rb") as f:
        magic      = f.read(6)
        major      = f.read(1)[0]
        minor      = f.read(1)[0]
        hlen       = struct.unpack("<H", f.read(2))[0]
        header_str = f.read(hlen).decode().strip()
        data_offset = f.tell()
    file_size      = os.path.getsize(path)
    elem_bytes     = arr.dtype.itemsize
    expected_data  = arr.size * elem_bytes
    expected_total = data_offset + expected_data
    size_ok        = file_size == expected_total
    print(f"  .npy header  :")
    print(f"    magic        : {magic}")
    print(f"    version      : {major}.{minor}")
    print(f"    header len   : {hlen} bytes  (data starts at byte {data_offset})")
    print(f"    header dict  : {header_str}")
    print(f"  Size check   :")
    print(f"    file size    : {file_size} bytes")
    print(f"    expected     : {data_offset} + {arr.size} x {elem_bytes} = {expected_total} bytes")
    print(f"    match        : {'YES' if size_ok else 'NO -- mismatch!'}")

    if arr.size == 0:
        print("  Values       : (empty array)")
        return

    # determine whether to display as bool or numeric
    unique_vals = np.unique(arr)
    is_binary = set(unique_vals.tolist()).issubset({0, 1, False, True})

    if is_binary:
        display = arr.astype(bool)
        print(f"  Values (bool representation, unique={unique_vals.tolist()}):")
    else:
        display = arr
        print(f"  Values (numeric, unique count={len(unique_vals)}):")

    # print full array via numpy (honour line-width)
    with np.printoptions(threshold=np.inf, linewidth=120):
        for line in str(display).splitlines():
            print(f"    {line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse AlphaQubit output .npy files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "file",
        metavar="*.npy",
        help="path to a .npy file",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"[ERROR] file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    analyze(args.file)


if __name__ == "__main__":
    main()
