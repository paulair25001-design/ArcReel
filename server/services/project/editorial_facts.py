"""Mechanical editing facts for the native Agent, without a quality verdict."""

from __future__ import annotations

from typing import Any

from lib.edit_timeline.readout import EditTimelineReadout


def timeline_editing_facts(readout: EditTimelineReadout) -> dict[str, Any]:
    """Expose holds, reused source ranges and narration spacing from one revision.

    Narration gaps are between complete audio files, not measured silence. A
    source range reused in another clip is observable; whether the repetition
    serves the story remains the native editor's decision.
    """
    clips = [clip for clip in readout.clips if clip.status == "ready" and clip.duration > 0]
    hold_seconds = sum(clip.hold for clip in clips)
    seen: dict[tuple[str, int], list[tuple[float, float, str]]] = {}
    repeated_ranges: list[dict[str, Any]] = []
    narration_gaps: list[dict[str, Any]] = []
    previous_narration: tuple[int, str, float] | None = None
    same_unit_transitions = different_unit_transitions = hard_cuts = 0
    for position, clip in enumerate(clips):
        if clip.video_version is not None and clip.source_duration is not None:
            start, end = 0.0, clip.source_duration
            if clip.trim is not None and clip.trim.basis_version == clip.video_version:
                start = min(clip.trim.source_in, end)
                end = max(start, min(clip.trim.source_out, end))
            previous = seen.setdefault((clip.unit_id, clip.video_version), [])
            overlaps = sorted(
                (max(start, left), min(end, right), clip_id)
                for left, right, clip_id in previous
                if left < end and right > start
            )
            cursor, overlap = start, 0.0
            for left, right, _ in overlaps:
                overlap += max(0.0, right - max(cursor, left))
                cursor = max(cursor, right)
            if overlap > 0:
                repeated_ranges.append(
                    {
                        "clip_id": clip.id,
                        "unit_id": clip.unit_id,
                        "version": clip.video_version,
                        "source_in": start,
                        "source_out": end,
                        "overlap_seconds": round(overlap, 6),
                        "previous_clip_ids": [clip_id for _, _, clip_id in overlaps],
                    }
                )
            previous.append((start, end, clip.id))
        if clip.narration is not None and clip.narration.end is not None:
            if previous_narration is not None:
                index, previous_clip, previous_end = previous_narration
                narration_gaps.append(
                    {
                        "from_clip_id": previous_clip,
                        "to_clip_id": clip.id,
                        "gap_seconds": round(clip.narration.start - previous_end, 6),
                        "intervening_clip_ids": [item.id for item in clips[index + 1 : position]],
                    }
                )
            previous_narration = position, clip.id, clip.narration.end
        if position + 1 < len(clips):
            following = clips[position + 1]
            if clip.transition_to_next is None:
                hard_cuts += 1
            elif clip.unit_id == following.unit_id:
                same_unit_transitions += 1
            else:
                different_unit_transitions += 1
    return {
        "assessment": "timing_facts_only",
        "hold_seconds": round(hold_seconds, 6),
        "hold_ratio": round(hold_seconds / readout.duration, 6) if readout.duration > 0 else 0.0,
        "replayed_source_seconds": round(sum(item["overlap_seconds"] for item in repeated_ranges), 6),
        "repeated_ranges": repeated_ranges,
        "transitions": {
            "same_unit": same_unit_transitions,
            "different_unit": different_unit_transitions,
            "hard_cuts": hard_cuts,
        },
        "narration_file_gaps": narration_gaps,
    }
