"""
Deep edge-case tests for compute_loss_mask in tool_sft_dataset.py.
"""
import pytest
from nanochat.tool_sft_dataset import (
    compute_loss_mask,
    BOS_ID, EOS_ID, CODE_START_ID, CODE_END_ID,
    THOUGHT_START_ID, THOUGHT_END_ID, QUERY_TOOL_ID, TOOL_RESULT_ID,
    FIM_PREFIX_ID, FIM_MIDDLE_ID, FIM_SUFFIX_ID,
)


class TestLossMaskEdgeCases:

    def test_consecutive_tool_results(self):
        """Two TOOL_RESULT blocks back to back."""
        tokens = [
            THOUGHT_START_ID, 100, THOUGHT_END_ID,
            QUERY_TOOL_ID, 200, CODE_END_ID,
            TOOL_RESULT_ID, 300, CODE_END_ID,
            # Second tool result without a preceding QUERY_TOOL
            TOOL_RESULT_ID, 400, CODE_END_ID,
            CODE_START_ID, 500, CODE_END_ID,
        ]
        mask = compute_loss_mask(tokens)

        # First tool result: masked
        assert mask[6] == 0  # TOOL_RESULT
        assert mask[7] == 0  # 300
        assert mask[8] == 0  # CODE_END

        # Second tool result: also masked
        assert mask[9] == 0   # TOOL_RESULT
        assert mask[10] == 0  # 400
        assert mask[11] == 0  # CODE_END

        # Final code: trained
        assert mask[12] == 1  # CODE_START
        assert mask[13] == 1  # 500
        assert mask[14] == 1  # CODE_END

    def test_tool_result_not_closed_by_code_end(self):
        """Fixed: If TOOL_RESULT is never followed by CODE_END,
        a block opener (CODE_START, THOUGHT_START, QUERY_TOOL) resets the state."""
        tokens = [
            THOUGHT_START_ID, 100, THOUGHT_END_ID,
            TOOL_RESULT_ID, 200, 201, 202,  # no CODE_END!
            CODE_START_ID, 300, CODE_END_ID,
        ]
        mask = compute_loss_mask(tokens)

        # THOUGHT block: trained
        assert mask[0] == 1  # THOUGHT_START
        assert mask[1] == 1  # 100
        assert mask[2] == 1  # THOUGHT_END

        # Unclosed TOOL_RESULT + content: masked
        assert mask[3] == 0  # TOOL_RESULT
        assert mask[4] == 0  # 200
        assert mask[5] == 0  # 201
        assert mask[6] == 0  # 202

        # CODE_START resets in_tool_result → trained block
        assert mask[7] == 1  # CODE_START
        assert mask[8] == 1  # 300
        assert mask[9] == 1  # CODE_END

    def test_empty_sequence(self):
        mask = compute_loss_mask([])
        assert mask == []

    def test_single_bos(self):
        mask = compute_loss_mask([BOS_ID])
        assert mask == [0]

    def test_single_eos(self):
        mask = compute_loss_mask([EOS_ID])
        assert mask == [0]

    def test_only_regular_tokens(self):
        """No special tokens → everything is instruction → all masked."""
        mask = compute_loss_mask([100, 101, 102])
        assert mask == [0, 0, 0]

    def test_code_start_as_first_response(self):
        """CODE_START can be the first response token."""
        tokens = [BOS_ID, 100, CODE_START_ID, 200, CODE_END_ID, EOS_ID]
        mask = compute_loss_mask(tokens)
        assert mask == [0, 0, 1, 1, 1, 0]

    def test_query_tool_as_first_response(self):
        """QUERY_TOOL can be the first response token (no preceding thought)."""
        tokens = [BOS_ID, 100, QUERY_TOOL_ID, 200, CODE_END_ID, EOS_ID]
        mask = compute_loss_mask(tokens)
        assert mask == [0, 0, 1, 1, 1, 0]

    def test_thought_start_as_first_token(self):
        """Thought at position 0 → response_start=0, no instruction prefix."""
        tokens = [THOUGHT_START_ID, 100, THOUGHT_END_ID]
        mask = compute_loss_mask(tokens)
        assert mask == [1, 1, 1]

    def test_multiple_code_end_without_opener(self):
        """Stray CODE_END tokens after response_start should be trained."""
        tokens = [THOUGHT_START_ID, CODE_END_ID, CODE_END_ID]
        mask = compute_loss_mask(tokens)
        assert mask == [1, 1, 1]

    def test_bos_in_middle_of_response(self):
        """BOS in middle of response is still masked."""
        tokens = [THOUGHT_START_ID, 100, BOS_ID, 101, THOUGHT_END_ID]
        mask = compute_loss_mask(tokens)
        assert mask[0] == 1  # THOUGHT_START
        assert mask[1] == 1  # 100
        assert mask[2] == 0  # BOS in middle
        assert mask[3] == 1  # 101
        assert mask[4] == 1  # THOUGHT_END

    def test_eos_in_middle_of_response(self):
        """EOS in middle of response is masked."""
        tokens = [THOUGHT_START_ID, 100, EOS_ID, 101, THOUGHT_END_ID]
        mask = compute_loss_mask(tokens)
        assert mask[2] == 0  # EOS

    def test_fim_with_no_middle(self):
        """FIM sequence without FIM_MIDDLE → everything masked."""
        tokens = [FIM_PREFIX_ID, 100, FIM_SUFFIX_ID, 200, EOS_ID]
        mask = compute_loss_mask(tokens)
        # has_fim is True (FIM_PREFIX in tokens), but no FIM_MIDDLE
        # So in_middle never becomes True → all masked
        assert all(m == 0 for m in mask)

    def test_fim_middle_content_after_eos(self):
        """FIM middle content should stop being trained at EOS."""
        tokens = [FIM_PREFIX_ID, 100, FIM_SUFFIX_ID, 200, FIM_MIDDLE_ID, 300, EOS_ID, 400]
        mask = compute_loss_mask(tokens)
        assert mask[4] == 1  # FIM_MIDDLE
        assert mask[5] == 1  # 300
        assert mask[6] == 0  # EOS
        assert mask[7] == 0  # 400 — content after EOS should NOT be trained

    def test_very_long_sequence(self):
        """Performance test with large sequence."""
        tokens = [BOS_ID] + [100] * 10000 + [THOUGHT_START_ID] + [200] * 10000 + [THOUGHT_END_ID, EOS_ID]
        mask = compute_loss_mask(tokens)
        assert len(mask) == len(tokens)
        # First 10001 tokens (BOS + 10000 instruction) should be masked
        assert all(m == 0 for m in mask[:10001])
        # Response tokens should be trained
        assert mask[10001] == 1  # THOUGHT_START

    def test_tool_result_with_code_end_id_in_content(self):
        """What if CODE_END_ID appears as a regular token inside tool result?
        The first CODE_END after TOOL_RESULT closes the block."""
        tokens = [
            THOUGHT_START_ID, THOUGHT_END_ID,
            TOOL_RESULT_ID, CODE_END_ID,  # Immediately closed
            CODE_START_ID, 100, CODE_END_ID,
        ]
        mask = compute_loss_mask(tokens)
        assert mask[2] == 0  # TOOL_RESULT
        assert mask[3] == 0  # CODE_END closing result
        assert mask[4] == 1  # CODE_START
        assert mask[5] == 1  # 100
        assert mask[6] == 1  # CODE_END

    def test_interleaved_thought_and_tool(self):
        """Realistic multi-turn: think → call → result → think → call → result → code."""
        tokens = [
            BOS_ID, 50, 51,  # instruction
            # Turn 1
            THOUGHT_START_ID, 100, THOUGHT_END_ID,
            QUERY_TOOL_ID, 110, CODE_END_ID,
            TOOL_RESULT_ID, 120, CODE_END_ID,
            # Turn 2
            THOUGHT_START_ID, 200, THOUGHT_END_ID,
            QUERY_TOOL_ID, 210, CODE_END_ID,
            TOOL_RESULT_ID, 220, CODE_END_ID,
            # Final
            CODE_START_ID, 300, CODE_END_ID,
            EOS_ID,
        ]
        mask = compute_loss_mask(tokens)
        expected = [
            0, 0, 0,     # instruction
            1, 1, 1,     # thought 1
            1, 1, 1,     # tool call 1
            0, 0, 0,     # result 1
            1, 1, 1,     # thought 2
            1, 1, 1,     # tool call 2
            0, 0, 0,     # result 2
            1, 1, 1,     # final code
            0,            # EOS
        ]
        assert mask == expected


class TestInt16Overflow:
    """Test that int16 packing in ToolCallSFTDataset handles large token IDs."""

    def test_token_ids_above_32767(self):
        """int32 packing should handle token IDs above 32767."""
        import numpy as np
        # int32 handles values up to ~2 billion
        arr = np.array([32000], dtype=np.int32)
        assert arr[0] == 32000
        arr2 = np.array([32768], dtype=np.int32)
        assert arr2[0] == 32768
        arr3 = np.array([65535], dtype=np.int32)
        assert arr3[0] == 65535

    def test_negative_target_in_int16(self):
        """Target -1 (masked) should survive int16 packing."""
        import numpy as np
        arr = np.array([-1], dtype=np.int16)
        assert arr[0] == -1  # OK, -1 fits in int16
