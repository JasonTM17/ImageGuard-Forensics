"""Entry point: `python -m imageguard`."""

from imageguard import __version__


def main() -> None:
    print(f"ImageGuard Forensics v{__version__}")
    print("Desktop UI available from Phase 7. See README.md for setup.")


if __name__ == "__main__":
    main()
