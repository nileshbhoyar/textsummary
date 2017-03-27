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
 python run.py --decode False -attention False --celltype GRU
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
 python run.py --decode True -attention False --celltype GRU
```
GRU without Attention
```
 python run.py --decode True -attention False --celltype GRU
```
