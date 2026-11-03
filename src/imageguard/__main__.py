"""Entry point: `python -m imageguard`."""

from imageguard import __version__


def main() -> None:
    print(f"ImageGuard Forensics v{__version__}")
    print("Desktop UI sẽ khả dụng từ Phase 7. Xem README.md để cài đặt.")


if __name__ == "__main__":
    main()
