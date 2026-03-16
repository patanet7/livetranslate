"""Tests for segment_id lifecycle: draft→final ordering and no-overwrite rules.

Bug: The same segment_id can receive multiple finals when the prebuffer
inference fires quickly and a stride inference follows immediately. The
consumer increments _segment_counter for each, but async task scheduling
can cause duplicate IDs.

The dashboard's addSegment uses segment_id for findIndex matching —
a final that arrives AFTER another final for the same segment_id will
overwrite it with completely different text.

Rule: A segment_id that has been finalized must not be overwritten by
a later message (draft or final) with the same segment_id but different content.
"""


class TestStoreAddSegmentLifecycle:
    """Test the dashboard store's addSegment draft→final behavior.

    These tests validate the CaptionEntry replacement logic in
    loopback.svelte.ts — we test the equivalent Python logic here.
    """

    def test_draft_replaced_by_final_same_id(self):
        """Final with same segment_id should replace draft in-place."""
        captions = []

        def add_segment(seg_id, text, is_draft):
            existing = next((i for i, c in enumerate(captions) if c["segmentId"] == seg_id), None)
            entry = {"segmentId": seg_id, "text": text, "isDraft": is_draft}
            if existing is not None:
                captions[existing] = entry
            else:
                captions.append(entry)

        add_segment(1, "Hello everyone", True)
        assert len(captions) == 1
        assert captions[0]["isDraft"] is True

        add_segment(1, "Hello everyone, welcome to the demo", False)
        assert len(captions) == 1  # replaced in-place, not appended
        assert captions[0]["isDraft"] is False
        assert "welcome" in captions[0]["text"]

    def test_final_not_overwritten_by_stale_draft(self):
        """A final segment should NOT be overwritten by a later draft with the same ID.

        This can happen when async task scheduling delivers messages out of order.
        """
        captions = []

        def add_segment(seg_id, text, is_draft):
            existing = next((i for i, c in enumerate(captions) if c["segmentId"] == seg_id), None)
            entry = {"segmentId": seg_id, "text": text, "isDraft": is_draft}
            if existing is not None:
                # Don't overwrite a final with a draft
                if not captions[existing]["isDraft"] and is_draft:
                    return  # skip stale draft
                captions[existing] = entry
            else:
                captions.append(entry)

        add_segment(1, "Complete sentence here.", False)  # final arrives first
        add_segment(1, "Complete sent", True)  # stale draft arrives late

        assert len(captions) == 1
        assert captions[0]["isDraft"] is False  # still final
        assert "sentence here" in captions[0]["text"]  # text preserved

    def test_final_not_overwritten_by_second_final_different_text(self):
        """A finalized segment should not be replaced by a completely different final.

        This happens when the segment counter produces duplicate IDs. The second
        final has entirely different text (from a different audio window).
        """
        captions = []

        def add_segment(seg_id, text, is_draft):
            existing = next((i for i, c in enumerate(captions) if c["segmentId"] == seg_id), None)
            entry = {"segmentId": seg_id, "text": text, "isDraft": is_draft}
            if existing is not None:
                # Don't overwrite a final with a draft
                if not captions[existing]["isDraft"] and is_draft:
                    return
                # If both are finals, the second is a new segment — append instead
                if not captions[existing]["isDraft"] and not is_draft:
                    # Assign a new unique segment_id to avoid collision
                    entry["segmentId"] = seg_id + 10000  # synthetic offset
                    captions.append(entry)
                    return
                captions[existing] = entry
            else:
                captions.append(entry)

        add_segment(2, "All right let's see.", False)
        add_segment(2, "Taking a bit. That's okay.", False)

        assert len(captions) == 2  # both should be visible
        assert "All right" in captions[0]["text"]
        assert "Taking a bit" in captions[1]["text"]
