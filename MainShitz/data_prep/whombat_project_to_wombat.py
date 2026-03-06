"""
whombat_project_to_wombat.py - Format converter from Whombat to Wombat

Whombat exports one big JSON with everything. Our pipeline wants
per-audio JSONs. This script does the translation.

Takes Whombat's nested structure and spits out clean per-recording JSONs
with start_time, end_time, and species labels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ConvertedProjectSummary:
    jsons_written: int
    recordings_seen: int
    sound_events_seen: int
    sound_events_written: int
    sound_events_skipped_unlabeled: int


def _safe_filename_stem(name: str):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip("._-") or "recording"


def _basename_from_any_path(path_str: str) -> str:
    #Return filename
    if '\\' in path_str or (len(path_str) >= 2 and path_str[1] == ':'):
        return PureWindowsPath(path_str).name
    return Path(path_str).name


def convert_whombat_project_to_wombat_jsons(
    project_json_path: str | Path,
    output_dir: str | Path,
    *,
    tag_key: str = "Species",
    skip_unlabeled: bool = True,
) -> ConvertedProjectSummary:
    """Convert a Whombat project export into a directory of per-recording JSONs.

    Args:
        project_json_path: Path to a Whombat project JSON export.
        output_dir: Directory to write per-recording Wombat-style JSON files.
        tag_key: Only tags whose "key" matches this will be used for labels.
        skip_unlabeled: If True, sound events without a label tag are skipped.

    Returns:
        A summary of what was written.
    """

    project_json_path = Path(project_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with project_json_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected Whombat JSON schema in {project_json_path}")

    # tag id -> label
    tag_id_to_label: Dict[int, str] = {}
    for tag in data.get("tags", []) or []:
        try:
            if tag.get("key") != tag_key:
                continue
            tag_id_to_label[int(tag["id"])] = str(tag.get("value", "")).strip()
        except Exception:
            continue

    rec_uuid_to_path: Dict[str, str] = {}
    rec_uuid_to_basename: Dict[str, str] = {}
    for rec in data.get("recordings", []) or []:
        uuid = rec.get("uuid")
        path = rec.get("path")
        if not uuid or not path:
            continue
        rec_uuid_to_path[str(uuid)] = str(path)
        rec_uuid_to_basename[str(uuid)] = _basename_from_any_path(str(path))

    se_uuid_to_info: Dict[str, Tuple[str, float, float, Optional[float], Optional[float]]] = {}
    for se in data.get("sound_events", []) or []:
        se_uuid = se.get("uuid")
        rec_uuid = se.get("recording")
        geom = se.get("geometry") or {}
        coords = geom.get("coordinates") if isinstance(geom, dict) else None
        if not se_uuid or not rec_uuid or not isinstance(coords, list) or len(coords) < 4:
            continue
        try:
            start_s = float(coords[0])
            low_hz = float(coords[1])
            end_s = float(coords[2])
            high_hz = float(coords[3])
        except Exception:
            continue
        se_uuid_to_info[str(se_uuid)] = (str(rec_uuid), start_s, end_s, low_hz, high_hz)

    # sound_event uuid -> list of tag ids
    se_uuid_to_tag_ids: Dict[str, List[int]] = {}
    for ann in data.get("sound_event_annotations", []) or []:
        se_uuid = ann.get("sound_event")
        tag_ids = ann.get("tags")
        if not se_uuid or not isinstance(tag_ids, list):
            continue
        cleaned: List[int] = []
        for tid in tag_ids:
            try:
                cleaned.append(int(tid))
            except Exception:
                continue
        if cleaned:
            se_uuid_to_tag_ids[str(se_uuid)] = cleaned

    # group annotations by uuid
    rec_uuid_to_annotations: Dict[str, List[Dict[str, Any]]] = {}
    skipped_unlabeled = 0
    written_events = 0

    for se_uuid, (rec_uuid, start_s, end_s, low_hz, high_hz) in se_uuid_to_info.items():
        tag_ids = se_uuid_to_tag_ids.get(se_uuid, [])
        label = None
        for tid in tag_ids:
            if tid in tag_id_to_label:
                label = tag_id_to_label[tid]
                break

        if not label:
            if skip_unlabeled:
                skipped_unlabeled += 1
                continue
            label = "unknown"

        ann: Dict[str, Any] = {
            "start_time": float(start_s),
            "end_time": float(end_s),
            "label": label,
        }
        if low_hz is not None and high_hz is not None:
            ann["low_freq_hz"] = float(low_hz)
            ann["high_freq_hz"] = float(high_hz)

        rec_uuid_to_annotations.setdefault(rec_uuid, []).append(ann)
        written_events += 1

    # write JSON
    jsons_written = 0
    for rec_uuid, anns in rec_uuid_to_annotations.items():
        audio_basename = rec_uuid_to_basename.get(rec_uuid)
        if not audio_basename:
            # fall back to uuid if recording list is incomplete
            audio_basename = f"{rec_uuid}.wav"

        out_stem = _safe_filename_stem(Path(audio_basename).stem)
        out_path = output_dir / f"{out_stem}.json"

        anns_sorted = sorted(anns, key=lambda a: (a.get("start_time", 0.0), a.get("end_time", 0.0)))

        payload = {
            "audio_file": audio_basename,
            # Keep as basename
            "recording": audio_basename,
            "annotations": anns_sorted,
            "source": {
                "type": "whombat_project",
                "project_json": str(project_json_path),
                "recording_uuid": rec_uuid,
                "original_recording_path": rec_uuid_to_path.get(rec_uuid),
            },
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        jsons_written += 1

    return ConvertedProjectSummary(
        jsons_written=jsons_written,
        recordings_seen=len(rec_uuid_to_path),
        sound_events_seen=len(se_uuid_to_info),
        sound_events_written=written_events,
        sound_events_skipped_unlabeled=skipped_unlabeled,
    )


def _main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Convert Whombat project export to per-audio Wombat JSONs")
    parser.add_argument("--project_json", required=True, help="Path to Whombat project export JSON")
    parser.add_argument("--out_dir", required=True, help="Output directory for per-audio JSONs")
    parser.add_argument("--tag_key", default="Species", help="Tag key to use as label")
    parser.add_argument("--keep-unlabeled", action="store_true", help="Do not skip unlabeled sound events")

    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = convert_whombat_project_to_wombat_jsons(
        args.project_json,
        args.out_dir,
        tag_key=args.tag_key,
        skip_unlabeled=not args.keep_unlabeled,
    )

    print(
        "Converted", args.project_json, "->", args.out_dir,
        f"(jsons_written={summary.jsons_written}, sound_events_written={summary.sound_events_written}, skipped_unlabeled={summary.sound_events_skipped_unlabeled})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
