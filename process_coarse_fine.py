def process_coarse_fine() -> None:
    import sys

    original_argv = list(sys.argv)
    try:
        # process_coarse_fine_dtc parses argv in its own CLI flow; keep only program name.
        sys.argv = [original_argv[0]]
        from process_coarse_fine_dtc import main as _process_main
        _process_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    process_coarse_fine()
