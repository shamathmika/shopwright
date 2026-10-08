from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from shopwright.config import EMBED_MODEL

OUT = Path(__file__).resolve().parent / "models" / "bge_small" / "1" / "model.onnx"


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        cls = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, dim=-1)


def main():
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    encoder = Encoder(AutoModel.from_pretrained(EMBED_MODEL)).eval()
    sample = tokenizer(["hello world"], return_tensors="pt")
    torch.onnx.export(
        encoder, (sample["input_ids"], sample["attention_mask"]), OUT,
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "embedding": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
