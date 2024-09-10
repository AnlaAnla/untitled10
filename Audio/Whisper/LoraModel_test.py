from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from datasets import Audio
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torchaudio
import os

from datasets import load_dataset, DatasetDict
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
import evaluate

from transformers import Seq2SeqTrainingArguments, BitsAndBytesConfig
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_name_or_path = "openai/whisper-large-v3"
    task = "transcribe"

    # 此处加载我的数据集
    dataset_name = r"D:\Code\ML\Audio\Data1"
    language = "Chinese"

    common_voice = DatasetDict()
    common_voice["test"] = load_dataset("audiofolder", data_dir=r"D:\Code\ML\Audio\Data1", split="train")

    print(common_voice)
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)


    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["test"], num_proc=1)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load("wer")

    peft_model_id = r"D:\Code\ML\Project\untitled10\Audio\Whisper\reach-vb\test\checkpoint-100" # Use the same model ID as before.
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.config.use_cache = True

    eval_dataloader = DataLoader(common_voice["test"], batch_size=8, collate_fn=data_collator)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    normalizer = BasicTextNormalizer()

    predictions = []
    references = []
    normalized_predictions = []
    normalized_references = []

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                print(step, decoded_preds)

                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
                normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])
                normalized_references.extend([normalizer(label).strip() for label in decoded_labels])
            del generated_tokens, labels, batch
        gc.collect()
    wer = 100 * metric.compute(predictions=predictions, references=references)
    normalized_wer = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
    eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer}

    print(f"{wer=} and {normalized_wer=}")
    print(eval_metrics)