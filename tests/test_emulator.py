#!/usr/bin/env python3
"""CLI for running Jarvis emulation scenarios.

Usage:
    python tests/test_emulator.py                         # Run all scenarios
    python tests/test_emulator.py --fast                  # Accelerated (100x)
    python tests/test_emulator.py --debug                 # Enable debug flags
    python tests/test_emulator.py tests/scenarios/X.yaml  # Single scenario
"""

import argparse
import glob
import os
import sys
import time

# Add project root to path so we can import the emulator package
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)

from emulator.runner import run_scenario


def discover_scenarios():
    """Find all YAML scenario files."""
    scenarios_dir = os.path.join(TESTS_DIR, "scenarios")
    return sorted(glob.glob(os.path.join(scenarios_dir, "*.yaml")))


def main():
    parser = argparse.ArgumentParser(description="Run Jarvis emulation scenarios")
    parser.add_argument("scenarios", nargs="*", help="Specific scenario files to run")
    parser.add_argument("--fast", action="store_true",
                        help="Run accelerated (time_scale=0.01)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable all debug flags")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed event logs")
    args = parser.parse_args()

    time_scale = 0.01 if args.fast else None

    if args.scenarios:
        scenario_files = args.scenarios
    else:
        scenario_files = discover_scenarios()

    if not scenario_files:
        print("No scenarios found.")
        sys.exit(1)

    print(f"Running {len(scenario_files)} scenario(s)...\n")

    passed = 0
    failed = 0
    total_time = 0

    for path in scenario_files:
        name = os.path.basename(path)
        t0 = time.perf_counter()

        try:
            report = run_scenario(path, time_scale=time_scale, debug=args.debug)
        except Exception as e:
            dt = time.perf_counter() - t0
            total_time += dt
            print(f"  FAIL  {name} ({dt:.1f}s) — exception: {e}")
            failed += 1
            continue

        dt = time.perf_counter() - t0
        total_time += dt

        if report.passed:
            print(f"  PASS  {report.name} ({dt:.1f}s)")
            passed += 1
        else:
            print(f"  FAIL  {report.name} ({dt:.1f}s)")
            for err in report.errors:
                print(f"        {err}")
            failed += 1

        if args.verbose:
            for event in report.events:
                print(f"        [{event['type']}] interaction={event['interaction_index']} {event['data']}")

    print(f"\nResults: {passed} passed, {failed} failed ({total_time:.1f}s total)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
