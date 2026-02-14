import json

from nanochat.tool_sft_dataset import ToolCallSFTDataset, compute_loss_mask


def test_compute_loss_mask_masks_tool_result_block():
    # <BOS> <THOUGHT_START> t <THOUGHT_END> <QUERY_TOOL> q <CODE_END>
    # <TOOL_RESULT> r r <CODE_END> <CODE_START> c <CODE_END> <EOS>
    token_ids = [2, 9, 101, 10, 11, 202, 8, 19, 303, 304, 8, 7, 404, 8, 3]
    mask = compute_loss_mask(token_ids)

    # Tool result block must be masked
    assert mask[7] == 0
    assert mask[8] == 0
    assert mask[9] == 0
    assert mask[10] == 0

    # Thought/tool-call/code blocks should be trained
    assert mask[1] == 1
    assert mask[4] == 1
    assert mask[11] == 1
    assert mask[12] == 1


def test_compute_loss_mask_uses_dynamic_special_ids():
    special_ids = {
        "bos": 1000,
        "eos": 1001,
        "code_start": 1002,
        "code_end": 1003,
        "thought_start": 1004,
        "thought_end": 1005,
        "query_tool": 1006,
        "tool_result": 1007,
        "fim_prefix": 1008,
        "fim_middle": 1009,
        "fim_suffix": 1010,
    }
    token_ids = [1000, 55, 1004, 66, 1005, 1006, 77, 1003, 1007, 88, 1003, 1002, 99, 1003, 1001]
    mask = compute_loss_mask(token_ids, special_ids=special_ids)

    assert mask[8] == 0
    assert mask[9] == 0
    assert mask[10] == 0
    assert mask[2] == 1
    assert mask[11] == 1


def test_tool_call_dataset_loads_text_schema(cpp_tokenizer_dir, monkeypatch):
    """Test ToolCallSFTDataset loads and tokenizes text schema using tokenizer.json."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(cpp_tokenizer_dir))
    monkeypatch.setenv("NANOCHAT_CPP_TOKENIZER", "1")

    sample = {
        "text": "<BOS>\n// task\n<THOUGHT_START>\n// think\n<THOUGHT_END>\n<CODE_START>\nint x = 1;\n<CODE_END>\n<EOS>",
        "source": "no_tool",
    }
    data_path = cpp_tokenizer_dir / "tool_small.jsonl"
    data_path.write_text(json.dumps(sample) + "\n")

    ds = ToolCallSFTDataset(str(data_path), tokenizer_name="cpp", max_len=256)
    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape == y.shape
    assert (y != -1).sum().item() > 0
