# sequence_embedder.py

import torch
import esm
from typing import List, Union


class ESMEmbedder:
    """
    ESMEmbedder: Extract residue-level embeddings using ESM model
    使用 ESM 模型提取残基级别嵌入向量

    Args:
        model_name (str): ESM 模型名称（默认使用 esm2_t33_650M_UR50D）
        device (str): 设备名称（默认自动检测，cuda 优先）
    """

    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    def embed(self, sequences: Union[str, List[str]]) -> torch.Tensor:
        """
        Extract residue-level ESM embeddings.
        提取残基级别的 ESM 嵌入表示

        Args:
            sequences: 单条序列（str）或多条序列（List[str]）

        Returns:
            Tensor: 单条返回 [L, D]；多条返回 [N, L, D]
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        data = [(f"seq{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        embeddings = []
        for i, (_, seq) in enumerate(data):
            emb = token_representations[i, 1:len(seq)+1].cpu()  # 去掉 <cls> 和 <eos>
            embeddings.append(emb)

        return embeddings[0] if len(embeddings) == 1 else embeddings

    def __call__(self, sequences: Union[str, List[str]]) -> torch.Tensor:
        return self.embed(sequences)


if __name__ == "__main__":
    embedder = ESMEmbedder(model_name="esm2_t33_650M_UR50D")
    seq = "MKTFFVIVAVLCLLSVAAQQEALAKEH"
    emb = embedder.embed(seq)
    print(emb.shape)  # [L, D]
    print(emb[:5])