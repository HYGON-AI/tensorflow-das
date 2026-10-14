# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Convert Bazel JSON Build Event Protocol test summaries to CI TSV files."""

import argparse
import json
import math
import os
import sys


SUCCESS_STATUSES = {"PASSED", "FLAKY"}
STATUS_PRIORITY = {
    "PASSED": 0,
    "FLAKY": 1,
    "NO_STATUS": 2,
    "INCOMPLETE": 3,
    "TIMEOUT": 4,
    "FAILED": 5,
    "REMOTE_FAILURE": 6,
    "FAILED_TO_BUILD": 7,
    "TOOL_HALTED_BEFORE_TESTING": 8,
    "NOT_REPORTED": 9,
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--bep", required=True)
  parser.add_argument("--selected-targets", required=True)
  parser.add_argument("--results", required=True)
  parser.add_argument("--failed-targets", required=True)
  return parser.parse_args()


def read_selected_targets(path):
  selected = []
  with open(path, encoding="utf-8") as selected_file:
    for line_number, line in enumerate(selected_file, start=1):
      line = line.rstrip("\n")
      if line_number == 1 and line == "suite\ttarget":
        continue
      if not line:
        continue
      try:
        suite, target = line.split("\t", 1)
      except ValueError as error:
        raise ValueError(
            f"invalid selected target at {path}:{line_number}: {line!r}"
        ) from error
      selected.append((suite, target))
  return selected


def duration_seconds(summary):
  duration_millis = summary.get("totalRunDurationMillis")
  if duration_millis is not None:
    return int(math.ceil(int(duration_millis) / 1000.0))

  duration = summary.get("totalRunDuration")
  if isinstance(duration, str) and duration.endswith("s"):
    return int(math.ceil(float(duration[:-1])))
  if isinstance(duration, dict):
    seconds = int(duration.get("seconds", 0))
    nanos = int(duration.get("nanos", 0))
    return seconds + int(nanos > 0)
  return 0


def worse_status(current, candidate):
  if current is None:
    return candidate
  if STATUS_PRIORITY.get(candidate, 100) > STATUS_PRIORITY.get(current, 100):
    return candidate
  return current


def read_bep(path, selected_targets):
  statuses = {}
  durations = {}
  decoded_events = 0
  if not os.path.exists(path):
    print(f"WARNING: Bazel BEP file was not created: {path}", file=sys.stderr)
    return statuses, durations

  with open(path, encoding="utf-8") as bep_file:
    for line_number, line in enumerate(bep_file, start=1):
      if not line.strip():
        continue
      try:
        event = json.loads(line)
      except json.JSONDecodeError as error:
        print(
            f"WARNING: ignoring invalid BEP JSON at {path}:{line_number}: {error}",
            file=sys.stderr,
        )
        continue
      decoded_events += 1

      event_id = event.get("id", {})
      summary_id = event_id.get("testSummary", {})
      label = summary_id.get("label")
      if label in selected_targets and "testSummary" in event:
        summary = event["testSummary"]
        status = summary.get("overallStatus", "NO_STATUS")
        statuses[label] = worse_status(statuses.get(label), status)
        durations[label] = max(durations.get(label, 0), duration_seconds(summary))
        continue

      completed_id = event_id.get("targetCompleted", {})
      label = completed_id.get("label")
      completed = event.get("completed", {})
      if (
          label in selected_targets
          and completed
          and not completed.get("success", False)
      ):
        statuses[label] = worse_status(
            statuses.get(label), "FAILED_TO_BUILD"
        )

  if decoded_events == 0:
    print(f"WARNING: Bazel BEP file contains no readable events: {path}", file=sys.stderr)
  return statuses, durations


def main():
  args = parse_args()
  selected = read_selected_targets(args.selected_targets)
  selected_set = {target for _, target in selected}
  statuses, durations = read_bep(args.bep, selected_set)

  with open(args.results, "w", encoding="utf-8", newline="") as results_file, open(
      args.failed_targets, "w", encoding="utf-8", newline=""
  ) as failed_file:
    results_file.write("suite\tstatus\telapsed_seconds\ttarget\n")
    for suite, target in selected:
      status = statuses.get(target, "NOT_REPORTED")
      results_file.write(
          f"{suite}\t{status}\t{durations.get(target, 0)}\t{target}\n"
      )
      if status not in SUCCESS_STATUSES:
        failed_file.write(f"{suite}\t{target}\n")


if __name__ == "__main__":
  main()
