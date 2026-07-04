#!/usr/bin/env python3

from pathlib import Path


def main() -> None:
    print("CCO eval scaffold")
    print(f"reports dir: {Path(__file__).resolve().parent / 'reports'}")


if __name__ == "__main__":
    main()
