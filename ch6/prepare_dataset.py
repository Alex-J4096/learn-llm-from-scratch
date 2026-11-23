import requests
import zipfile
import os
from pathlib import Path
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

# 创建一个平衡数据集
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    # 随机采样非垃圾消息使其数量与垃圾消息一致
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_dataset = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_dataset

def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df)*train_frac)
    validation_end = train_end + int(len(df)*validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 如果超过就截断
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        # 填充到最大长度
        self.encoded_texts = [
            encoded_text + [pad_token_id]*(self.max_length-len(encoded_text))
            for encoded_text in self.encoded_texts
        ]


    def _longest_encoded_length(self):
        max_len = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_len:
                max_len = encoded_length
        return max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

if __name__ == "__main__":
    # try:
    #     download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    # except (requests.exceptions.RequestException, TimeoutError) as e:
    #     print(f"Primary URL failed: {e}. Trying backup URL...")
    #     url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
    #     download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    # df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    # print(df)
    #
    # print(df["Label"].value_counts())
    #
    # balanced_df = create_balanced_dataset(df)
    # print(balanced_df["Label"].value_counts())
    #
    # balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    # train_df, validation_df, test_df = random_split(balanced_df, train_frac=0.7, validation_frac=0.1)
    #
    # train_df.to_csv("train.csv", index=None)
    # validation_df.to_csv("validation.csv", index=None)
    # test_df.to_csv("test.csv", index=None)

    tokenizer = tiktoken.get_encoding("gpt2")
    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)
    print(train_dataset.max_length)

    validate_dataset = SpamDataset(csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
    test_dataset = SpamDataset(csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)

    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    validation_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        drop_last=False
    )

    for input_batch, target_batch in train_loader:
        pass
    print("Input batch dimensions:", input_batch.shape)
    print("Target batch dimensions:", target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(validation_loader)} validation batches")
    print(f"{len(test_loader)} test batches")