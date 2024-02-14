import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from transformers import BertModel, BertTokenizer

logger = logging.getLogger("omunet")


def create_node_embeddings(processed_path: Path):
    device = get_cuda_device() if torch.cuda.is_available() else "cpu"

    logger.info("\tI will use the following device %s", device)
    tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

    for nodes_file_path in processed_path.glob("**/nodes.parquet"):
        onto_name = Path(nodes_file_path).parent.name
        embeddings_path = Path(nodes_file_path.parent, "embeddings.parquet")

        df = pd.read_parquet(nodes_file_path)
        if embeddings_path.exists() or "embeddings" in df.columns:
            logger.info("\tSkipping %s", onto_name)
            continue

        logger.info("\tCreating embeddings for %s", onto_name)

        embeddings = []
        token_ids = []

        labels_per_batch = 1000
        labels = df["labels"].map(lambda x: x[0]).values
        labels = np.array_split(labels, len(labels) / labels_per_batch)

        with tqdm.tqdm(total=len(df["labels"])) as progress_bar:
            for label_batch in labels:
                encoded_input = tokenizer(
                    label_batch.tolist(),
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    batch_embeddings = (
                        model(**encoded_input)
                        .pooler_output.detach()
                        .to("cpu")
                        .detach()
                        .numpy()
                    )
                torch.cuda.empty_cache()

                batch_tokens_ids = [
                    [
                        y for y in x if y and y not in [101, 102]
                    ]  # filter None values created by padding
                    for x in np.ma.masked_array(
                        encoded_input["input_ids"].cpu().numpy(),
                        1 - encoded_input["attention_mask"].cpu().numpy(),
                    ).tolist()
                ]
                embeddings.append(batch_embeddings)
                token_ids.extend(batch_tokens_ids)
                progress_bar.update(len(batch_embeddings))

        new_df = pd.DataFrame()
        new_df["id"] = df.id
        new_df["embeddings"] = pd.Series(np.vstack(embeddings).tolist())
        new_df["token_ids"] = pd.Series(token_ids)

        new_df.to_parquet(embeddings_path)


def get_cuda_device():
    devices = range(torch.cuda.device_count())
    memory_usage_of_devices = [torch.cuda.mem_get_info(x)[1] for x in devices]
    minimum_memory_usage = min(memory_usage_of_devices)
    return f"cuda:{memory_usage_of_devices.index(minimum_memory_usage)}"
