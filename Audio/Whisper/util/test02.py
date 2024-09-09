import torchaudio

dataset_name = "mozilla-foundation/common_voice_13_0"
language = "Hindi"
language_abbr = "hi" # Short hand code for the language we want to fine-tune

from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", use_auth_token=True, trust_remote_code=True)

print(common_voice)

common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant"]
)

print(common_voice)
print(common_voice["train"][0])