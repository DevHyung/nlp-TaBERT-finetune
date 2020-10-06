# TaBERT_finetune

## 사용법 Summary
```bash
1. tabert github repo를 clone
    1-1. tabert conda env 와 패키지들 설치
    (Installation 부분)

2. tabert pre-trained model down
    (Pre-trained Models 부분)

3. tabert 코드 구동 확인
    (Using a Pre_Trained Model 부분)

4. 이 Repo에 있는 all.json, querys.txt, train_finetune.py 를 taber Repo에 넣고 실행
    4-1. 우선은 모델 Config나 PATH관련 변수는 if __main__ 쪽에 하드코딩으로 구현
```

## Installation

First, clone the `tabert` github repo.
```bash
git clone https://github.com/facebookresearch/TaBERT.git
```

Second, install the conda environment `tabert` with supporting libraries.

```bash
bash scripts/setup_env.sh
```

Once the conda environment is created, install `TaBERT` using the following command:

```bash
conda activate tabert
pip install --editable .
```

**Integration with HuggingFace's pytorch-transformers Library** is still WIP. While all the pre-trained models were developed based on the old version of the library `pytorch-pretrained-bert`, they are compatible with the the latest version `transformers`. The conda environment will install both versions of the transformers library, and `TaBERT` will use `pytorch-pretrained-bert` by default. You could uninstall the `pytorch-pretrained-bert` library if you prefer using `TaBERT` with the latest version of `transformers`.

## Pre-trained Models

Pre-trained models could be downloaded from this [Google Drive shared folder](https://drive.google.com/drive/folders/1fDW9rLssgDAv19OMcFGgFJ5iyd9p7flg?usp=sharing).
Please uncompress the tarball files before usage.

Pre-trained models could be downloaded from command line as follows:
```shell script
pip install gdown

# TaBERT_Base_(K=3) -> 우리가 baseline 으로 잡고 있는 모델
gdown 'https://drive.google.com/uc?id=1NPxbGhwJF1uU9EC18YFsEZYE-IQR7ZLj'

# gdown 후에 다운받은 알집파일 해제 
tar -xvzf PATH/TO/MODEL.tar.gz
```

## Using a Pre-trained Model

To load a pre-trained model from a checkpoint file:

```python
from table_bert import TableBertModel

model = TableBertModel.from_pretrained(
    'path/to/pretrained/model/checkpoint.bin',
)
```

To produce representations of natural language text and and its associated table:
```python
from table_bert import Table, Column

table = Table(
    id='List of countries by GDP (PPP)',
    header=[
        Column('Nation', 'text', sample_value='United States'),
        Column('Gross Domestic Product', 'real', sample_value='21,439,453')
    ],
    data=[
        ['United States', '21,439,453'],
        ['China', '27,308,857'],
        ['European Union', '22,774,165'],
    ]
).tokenize(model.tokenizer)

# To visualize table in an IPython notebook:
# display(table.to_data_frame(), detokenize=True)

context = 'show me countries ranked by GDP'

# model takes batched, tokenized inputs
context_encoding, column_encoding, info_dict = model.encode(
    contexts=[model.tokenizer.tokenize(context)],
    tables=[table]
)
```

For the returned tuple, `context_encoding` and `column_encoding` are PyTorch tensors 
representing utterances and table columns, respectively. `info_dict` contains useful 
meta information (e.g., context/table masks, the original input tensors to BERT) for 
downstream application.

```python
context_encoding.shape
>>> torch.Size([1, 7, 768])

column_encoding.shape
>>> torch.Size([1, 2, 768])
```

**Use Vanilla BERT** To initialize a TaBERT model from the parameters of BERT:

```python
from table_bert import TableBertModel

model = TableBertModel.from_pretrained('bert-base-uncased')
```
## 출처
[TaBERT Github](https://github.com/facebookresearch/TaBERT)

## License

TaBERT is CC-BY-NC 4.0 licensed as of now.
