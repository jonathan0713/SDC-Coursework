"""
CLI tool for inspecting a debug log produced by DebugLogger.

Subcommands
───────────
  frame   --log FILE --frame N
      Print full state for frame N: tracks, detections, cost matrix, assignments.

  track   --log FILE --track ID [--frames A-B]
      Follow a single track across all (or a range of) frames.

  summary --log FILE
      Print per-frame counts (tracks, detections, matched, IDs present).

Usage
─────
  python src/tools/inspect_debug_log.py frame   --log debug.jsonl --frame 42
  python src/tools/inspect_debug_log.py track   --log debug.jsonl --track 5
  python src/tools/inspect_debug_log.py track   --log debug.jsonl --track 5 --frames 40-60
  python src/tools/inspect_debug_log.py summary --log debug.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_log(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _fmt_cost(v) -> str:
    if v is None:
        return "  ----"
    return f"{v:6.2f}"


# ── subcommand: frame ─────────────────────────────────────────────────────────

def cmd_frame(args):
    records = _load_log(args.log)
    frame_map = {r["frame"]: r for r in records}

    if args.frame not in frame_map:
        available = sorted(frame_map)
        print(f"Frame {args.frame} not found. Available: {available[0]}–{available[-1]}")
        sys.exit(1)

    r = frame_map[args.frame]
    print(f"\n{'═'*60}")
    print(f"  Frame {r['frame']}")
    print(f"{'═'*60}")

    tracks = r["tracks"]
    dets   = r["detections"]

    def _fv(v) -> str:
        return f"{v:6.2f}" if v is not None else "  N/A "

    print(f"\nTracks ({len(tracks)}):")
    print(f"  {'ID':>4}  {'state':>10}  {'hits':>4}  {'age':>3}  "
          f"{'pred_x':>7}  {'pred_y':>7}  {'vx':>7}  {'vy':>7}  {'rr':>6}")
    for t in tracks:
        print(f"  {t['track_id']:>4}  {t['state']:>10}  {t['hits']:>4}  {t['age']:>3}  "
              f"  {t['pred_pos'][0]:6.2f}   {t['pred_pos'][1]:6.2f}  "
              f" {_fv(t['velocity'][0])}  {_fv(t['velocity'][1])}  {t['last_range_rate']:6.2f}")

    print(f"\nDetections ({len(dets)}):")
    print(f"  {'idx':>4}  {'x':>7}  {'y':>7}  {'range_rate':>10}  {'cluster':>7}")
    for d in dets:
        print(f"  {d['det_idx']:>4}  {d['x']:7.2f}  {d['y']:7.2f}  "
              f"{d['range_rate']:>10.2f}  {d['cluster_id']:>7}")

    matched_set = {(ti, di) for ti, di in r["matched"]}

    if tracks and dets:
        print(f"\nCost matrix  (rows=tracks t0…t{len(tracks)-1}, "
              f"cols=dets d0…d{len(dets)-1}):")
        track_ids = [t["track_id"] for t in tracks]
        header = "  TID   " + "".join(f"  d{i:<3}" for i in range(len(dets)))
        print(header)
        cm = r["cost_matrix"]
        for ti, row in enumerate(cm):
            tid = track_ids[ti]
            cells = []
            for di, v in enumerate(row):
                marker = "*" if (ti, di) in matched_set else " "
                cells.append(marker + _fmt_cost(v))
            print(f"  t{tid:<4}" + "".join(cells))
        print("  (* = matched pair)")

    print(f"\nAssignments:")
    print(f"  matched          : {r['matched']}")
    print(f"  unmatched_tracks : {r['unmatched_tracks']}")
    print(f"  unmatched_dets   : {r['unmatched_dets']}")
    print(f"  valid_pairs      : {r['valid_pairs']}")
    print()


# ── subcommand: track ─────────────────────────────────────────────────────────

def cmd_track(args):
    records = _load_log(args.log)

    frame_lo, frame_hi = 0, math.inf
    if args.frames:
        parts = args.frames.split("-")
        if len(parts) == 2:
            frame_lo = int(parts[0])
            frame_hi = int(parts[1])
        else:
            frame_lo = frame_hi = int(parts[0])

    tid = args.track
    def _fv(v) -> str:
        return f"{v:6.2f}" if v is not None else "  N/A "

    print(f"\nTrack {tid} trajectory")
    print(f"{'frame':>6}  {'state':>10}  {'hits':>4}  {'age':>3}  "
          f"{'pred_x':>7}  {'pred_y':>7}  {'vx':>7}  {'vy':>7}  {'rr':>6}  assignment")

    found = False
    for r in records:
        f = r["frame"]
        if f < frame_lo or f > frame_hi:
            continue
        for t in r["tracks"]:
            if t["track_id"] != tid:
                continue
            found = True
            ti = next((i for i, tr in enumerate(r["tracks"]) if tr["track_id"] == tid), None)
            assign = ""
            if ti is not None:
                for (mti, mdi) in r["matched"]:
                    if mti == ti:
                        d = r["detections"][mdi]
                        assign = f"→ det[{mdi}] (x={d['x']:.2f}, rr={d['range_rate']:.2f})"
                        break
                if not assign and ti in r["unmatched_tracks"]:
                    assign = "unmatched"
            print(f"  {f:>4}  {t['state']:>10}  {t['hits']:>4}  {t['age']:>3}  "
                  f"  {t['pred_pos'][0]:6.2f}   {t['pred_pos'][1]:6.2f}  "
                  f" {_fv(t['velocity'][0])}  {_fv(t['velocity'][1])}  "
                  f"{t['last_range_rate']:6.2f}  {assign}")

    if not found:
        print(f"  (track {tid} not seen in the requested range)")
    print()


# ── subcommand: summary ───────────────────────────────────────────────────────

def cmd_summary(args):
    records = _load_log(args.log)
    print(f"\n{'frame':>6}  {'tracks':>6}  {'dets':>5}  {'matched':>7}  "
          f"{'unm_t':>6}  {'unm_d':>6}  track_ids")
    for r in records:
        ids = sorted({t["track_id"] for t in r["tracks"]})
        id_str = str(ids) if len(ids) <= 8 else f"{ids[:8]}…"
        print(f"  {r['frame']:>4}  {len(r['tracks']):>6}  {len(r['detections']):>5}  "
              f"{len(r['matched']):>7}  {len(r['unmatched_tracks']):>6}  "
              f"{len(r['unmatched_dets']):>6}  {id_str}")
    print()


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect a tracker debug log (.jsonl).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("frame", help="Dump full state for one frame")
    pf.add_argument("--log",   required=True)
    pf.add_argument("--frame", required=True, type=int)

    pt = sub.add_parser("track", help="Follow a single track across frames")
    pt.add_argument("--log",    required=True)
    pt.add_argument("--track",  required=True, type=int)
    pt.add_argument("--frames", default=None,
                    help="Frame range, e.g. 40-60  (inclusive)")

    ps = sub.add_parser("summary", help="Per-frame counts for all frames")
    ps.add_argument("--log", required=True)

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "frame":
        cmd_frame(args)
    elif args.cmd == "track":
        cmd_track(args)
    elif args.cmd == "summary":
        cmd_summary(args)


if __name__ == "__main__":
    main()
