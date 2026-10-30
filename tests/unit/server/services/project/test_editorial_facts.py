"""Native editing facts expose replay and timing without inventing approval."""

from __future__ import annotations

import pytest

from lib.edit_timeline.readout import ClipView, EditTimelineReadout, NarrationView, TimelineIdentity, TrimView
from server.services.project.editorial_facts import timeline_editing_facts


def _clip(clip_id, unit_id, *, start=0.0, duration=5.0, **changes):
    return ClipView(
        **{
            "id": clip_id,
            "unit_id": unit_id,
            "status": "ready",
            "start": start,
            "duration": duration,
            "video_version": 1,
            "source_duration": 5.0,
            "trim": None,
            "source_volume": 0.0,
            "hold": 0.0,
            "carries_narration": False,
            "narration": None,
            "reason": None,
            "transition_to_next": None,
            **changes,
        }
    )


def _readout(*clips):
    return EditTimelineReadout(
        timeline=TimelineIdentity(id="tl-demo", name="Edit", episode=3),
        revision=2,
        latest_revision=2,
        duration=max((clip.start + clip.duration for clip in clips), default=0.0),
        clips=clips,
        bgm=(),
        issues=(),
    )


def test_reused_head_exposes_its_full_range_and_separates_loop_dissolve_from_scene_join():
    readout = _readout(
        _clip(
            "c1",
            "S1",
            duration=2.7,
            trim=TrimView(source_in=0, source_out=2.7, basis_version=1),
            transition_to_next={"type": "dissolve", "duration": 0.5},
        ),
        _clip("c2", "S1", start=2.7, duration=3.74, trim=TrimView(source_in=0, source_out=3.74, basis_version=1)),
        _clip("c3", "S2", start=6.44),
    )
    before = readout.model_dump()
    facts = timeline_editing_facts(readout)

    assert facts["repeated_ranges"] == [
        {
            "clip_id": "c2",
            "unit_id": "S1",
            "version": 1,
            "source_in": 0.0,
            "source_out": 3.74,
            "overlap_seconds": 2.7,
            "previous_clip_ids": ["c1"],
        }
    ]
    assert facts["transitions"] == {"same_unit": 1, "different_unit": 0, "hard_cuts": 1}
    assert facts["assessment"] == "timing_facts_only"
    assert readout.model_dump() == before


def test_multiple_prior_uses_count_the_union_of_replayed_seconds():
    facts = timeline_editing_facts(
        _readout(
            _clip("c1", "S1", duration=3, trim=TrimView(source_in=0, source_out=3, basis_version=1)),
            _clip("c2", "S1", start=3, duration=3, trim=TrimView(source_in=0, source_out=3, basis_version=1)),
            _clip("c3", "S1", start=6, duration=3, trim=TrimView(source_in=1, source_out=4, basis_version=1)),
        )
    )
    assert facts["replayed_source_seconds"] == 5
    assert [item["overlap_seconds"] for item in facts["repeated_ranges"]] == [3, 2]


def test_disjoint_ranges_and_different_versions_are_not_replayed_source():
    facts = timeline_editing_facts(
        _readout(
            _clip("c1", "S1", duration=2, trim=TrimView(source_in=0, source_out=2, basis_version=1)),
            _clip("c2", "S1", start=2, duration=3, trim=TrimView(source_in=2, source_out=5, basis_version=1)),
            _clip("c3", "S1", start=5, video_version=2),
        )
    )
    assert facts["repeated_ranges"] == []
    assert facts["replayed_source_seconds"] == 0


def test_ignored_trim_is_counted_as_the_whole_current_source():
    facts = timeline_editing_facts(
        _readout(
            _clip("c1", "S1", duration=2, trim=TrimView(source_in=0, source_out=2, basis_version=1)),
            _clip("c2", "S1", start=2, trim=TrimView(source_in=3, source_out=4, basis_version=2)),
        )
    )
    [reused] = facts["repeated_ranges"]
    assert (reused["source_in"], reused["source_out"], reused["overlap_seconds"]) == (0, 5, 2)


def test_long_hold_and_file_spacing_do_not_mislabel_intervening_dialogue_as_silence():
    facts = timeline_editing_facts(
        _readout(
            _clip(
                "narration-1",
                "S1",
                duration=13.6,
                source_duration=4,
                hold=9.6,
                carries_narration=True,
                narration=NarrationView(start=0, end=13.6),
            ),
            _clip("dialogue", "S2", start=13.6, duration=8, source_duration=8, source_volume=1),
            _clip(
                "narration-2",
                "S3",
                start=21.6,
                duration=4,
                source_duration=4,
                carries_narration=True,
                narration=NarrationView(start=21.6, end=25.6),
            ),
        )
    )
    assert facts["hold_seconds"] == 9.6
    assert facts["hold_ratio"] == pytest.approx(9.6 / 25.6, abs=1e-6)
    assert facts["narration_file_gaps"] == [
        {
            "from_clip_id": "narration-1",
            "to_clip_id": "narration-2",
            "gap_seconds": 8.0,
            "intervening_clip_ids": ["dialogue"],
        }
    ]


def test_empty_timeline_has_no_false_transition_or_repetition():
    facts = timeline_editing_facts(_readout())
    assert facts["hold_ratio"] == 0
    assert facts["transitions"] == {"same_unit": 0, "different_unit": 0, "hard_cuts": 0}
    assert facts["repeated_ranges"] == facts["narration_file_gaps"] == []
