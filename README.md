# W266- Seq2seq Abstract Summarization
We have implemented Recurrent Neural Network (LSTM and GRU units) for developing Seq2Seq Encoder Decoded model with and without attention mechanism for
summarization of amazon food reviews into abstractive tips.
## Data processing.
Amazon reviews can be downloaded from
<a href="https://snap.stanford.edu/data/web-FineFoods.html">here</a>
first download dataset and unizp that file in folder data. if data folder does not exist then you will have to create it in your repo.

#### Data filtering and Indexing
Once data is downloaded then execute data processing using command
```
 python data_utils/ourmodel/data_util.py
```
## Execute Training

Run model using run.py as below to

Train LSTM with Attention
```
 python run.py --decode false -attention True --celltype LSTM
```
Train GRU with Attention
```
 python run.py --decode False --attention True --celltype GRU
```
Train LSTM without Attention
```
 python run.py --decode False -attention False --celltype LSTM
```
Train GRU without Attention
```
 python run.py --decode False -attention False --celltype GRU
```
## Running Summarization Interactively

You can test sumamrization for trained models using below commands.
 LSTM with Attention
```
 python run.py --decode True -attention True --celltype LSTM
```
GRU with Attention
```
 python run.py --decode True --attention True --celltype GRU
```
LSTM without Attention
```
 python run.py --decode True -attention False --celltype LSTM
```
GRU without Attention
```
 python run.py --decode True -attention False --celltype GRU
```

## Running Test Files to Calculate Rogue-1, Bleu-1 and F1 Scores
Before running, make sure you already created results folder
 LSTM with Attention
```
python run.py --decode False --self_test True --celltype LSTM --attention True
```
GRU with Attention
```
python run.py --decode False --self_test True --celltype GRU --attention True
```
LSTM without Attention
```
python run.py --decode False --self_test True --celltype LSTM --attention False
```
GRU without Attention
```
python run.py --decode False --self_test True --celltype GRU --attention False
```
## Example of model training

[![LSTMwithAttentionModelTraining.png](https://s22.postimg.org/isl6ucd0h/LSTMwith_Attention_Model_Training.png)](https://postimg.org/image/t2nltl2vx/)

We have also provided Ipython notebooks that gives some introduction to what is there in all .py files.
