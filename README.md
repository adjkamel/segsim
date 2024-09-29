This repository contains code for the paper:  [Action Segmentation via Iterative Coarse-to-Fine Similarity Measurement and Boundary
Correction](https://) 

## Environment

* Python 3.8.2
* PyTorch 3.9.16
* CUDA 12.0

## Data

The features, annotations, groundtruth, results, mapping, and checkpoint files must follow this hierarchy:


```
-50salads/                       -gtea/                     -breakfast/
    -features/                     -features/                  -features/
    -groundTruth/                  -groundTruth/               -groundTruth/
    -splits/                       -splits/                    -splits/
    -mapping.txt                   -mapping.txt                -mapping.txt
    -results/                      -results/                   -results/
      -/split1                       -/split1                    -/split1
        -fusion                        -fusion                     -fusion
      ...                            ...                         ...
      -/split5                       -/split4                    -/split4
        -fusion                        -fusion                     -fusion
    -checkpoints                   -checkpoints               
      -/split1                       -/split1                   
        -/32                           -/32
        -/64                           -/64
        -/128                          -/128
        -/256                          -/256
      ...                            ...                           
      -/split5                       -/split4
        -/32                           -/32
        -/64                           -/64
        -/128                          -/128
        -/256                          -/256                
```

Features of the  the three datasets can be downkoad from this repository: (here](https://github.com/yabufarha/ms-tcn))  

https://github.com/yabufarha/ms-tcn

## Supervised action segmentation:

### Training



* 50Salads  

Train the four encodres (32,64,128, and 256) for each split using:

```
python main.py -algo sup -t 32 -a train -d 50salads -sp 1 -gpu 0

python main.py -algo sup -t 64 -a train -d 50salads -sp 1 -gpu 0

python main.py -algo sup -t 128 -a train -d 50salads -sp 1 -gpu 0

python main.py -algo sup -t 256 -a train -d 50salads -sp 1 -gpu 0

python main.py -algo sup -t 32 -a train -d 50salads -sp 2 -gpu 0

....

python main.py -algo sup -t 256 -a train -d 50salads -sp 5 -gpu 0

```


After training the encoders, the fusion model can be trained with loading the best check points number given as argument to -ck32,-ck64,-ck128, and -ck256 of the four encoders using:


```
python main.py -algo sup -t fusion -a train -d 50salads -sp 1 -gpu 0 -bw 16 -sw 4 -ck32 94 -ck64 67 -ck128 90 -ck256 77

python main.py -algo sup -t fusion -a train -d 50salads -sp 2 -gpu 0 -bw 16 -sw 4 -ck32 96 -ck64 99 -ck128 100 -ck256 67
  
python main.py -algo sup -t fusion -a train -d 50salads -sp 3 -gpu 0 -bw 16 -sw 4 -ck32 78 -ck64 98 -ck128 96 -ck256 80

python main.py -algo sup -t fusion -a train -d 50salads -sp 4 -gpu 0 -bw 16 -sw 4 -ck32 88 -ck64 76 -ck128 91 -ck256 40
                     
python main.py -algo sup -t fusion -a train -d 50salads -sp 5 -gpu 0 -bw 16 -sw 4 -ck32 52 -ck64 93 -ck128 60 -ck256 16
```

* gtea

Same as 50Salads with 4 splits:

```
python main.py -algo sup -t 32 -a train -d gtea -sp 1 -gpu 0

python main.py -algo sup -t 64 -a train -d gtea -sp 1 -gpu 0

python main.py -algo sup -t 128 -a train -d gtea -sp 1 -gpu 0

python main.py -algo sup -t 256 -a train -d gtea -sp 1 -gpu 0

python main.py -algo sup -t 32 -a train -d gtea -sp 2 -gpu 0

...

python main.py -algo sup -t 256 -a train -d gtea -sp 5 -gpu 0

```
Train the fusion model with: 

```

python main.py -algo sup -t fusion -a train -d gtea -sp 1 -gpu 0 -bw 8 -sw 4 -ck32 39 -ck64 64 -ck128 64 -ck256 51

python main.py -algo sup -t fusion -a train -d gtea -sp 2 -gpu 1 -bw 8 -sw 4 -ck32 98 -ck64 74 -ck128 62 -ck256 59

python main.py -algo sup -t fusion -a train -d gtea -sp 3 -gpu 0 -bw 8 -sw 4 -ck32 83 -ck64 41 -ck128 75 -ck256 48

python main.py -algo sup -t fusion -a train -d gtea -sp 4 -gpu 1 -bw 8 -sw 4 -ck32 76 -ck64 51 -ck128 68 -ck256 40

```


2. GTEA



### Testing

The pretrained models of the encoders and the fusion model for all the splits of  the two datasets are provided [here](https://drive.google.com/drive/folders/1KYZImXp1DajgWc-cKUQEgsXaO8I1Afk0?usp=drive_link), copy them in the checkpoint dataset folders. The checkpoints of the encoders can be used to train the fusion model as illustrated previously. To test the models, run the following: 


* 50Salads  

```
python main.py -algo sup -t fusion -a predict -d 50salads -sp 1 -gpu 0 -bw 16 -sw 4 -ck32 94 -ck64 67 -ck128 90 -ck256 77 -ckfuse 15 -sthw 80 -bndin -bndaft 

python main.py -algo sup -t fusion -a predict -d 50salads -sp 2 -gpu 0 -bw 16 -sw 4 -ck32 96 -ck64 99 -ck128 100 -ck256 67 -ckfuse 22 -sthw 80 -bndin -bndaft
  
python main.py -algo sup -t fusion -a predict -d 50salads -sp 3 -gpu 0 -bw 16 -sw 4 -ck32 78 -ck64 98 -ck128 96 -ck256 80 -ckfuse 11 -sthw 80 -bndin -bndaft
 
python main.py -algo sup -t fusion -a predict -d 50salads -sp 4 -gpu 0 -bw 16 -sw 4 -ck32 88 -ck64 76 -ck128 91 -ck256 40 -ckfuse 17 -sthw 80 -bndin -bndaft
                     
python main.py -algo sup -t fusion -a predict -d 50salads -sp 5 -gpu 0 -bw 16 -sw 4 -ck32 52 -ck64 93 -ck128 60 -ck256 16 -ckfuse 30 -sthw 80 -bndin -bndaft
```

* GTEA  

```
python main.py -algo sup -t fusion -a predict -d gtea -sp 1 -gpu 0 -bw 8 -sw 4 -ck32 39 -ck64 64 -ck128 64 -ck256 51 -ckfuse 31 -sthw 4 -bndin

python main.py -algo sup -t fusion -a predict -d gtea -sp 2 -gpu 1 -bw 8 -sw 4 -ck32 98 -ck64 74 -ck128 62 -ck256 59 -ckfuse 79 -sthw 4 -bndin

python main.py -algo sup -t fusion -a predict -d gtea -sp 3 -gpu 0 -bw 8 -sw 4 -ck32 83 -ck64 41 -ck128 75 -ck256 48 -ckfuse 12 -sthw 4 -bndin

python main.py -algo sup -t fusion -a predict -d gtea -sp 4 -gpu 1 -bw 8 -sw 4 -ck32 76 -ck64 51 -ck128 68 -ck256 40 -ckfuse 63 -sthw 4 -bndin
```


## Unsupervised action segmentation:

1. 50Salads  and gtea

Based on the proposed unsupervised algorithm, it doesn't require training, run the following commands to get the prediction n the testing sets, The interval_between_boundaries is set to 500, 70, and 300 for 50Salads, gtea, and breakfast datasets, respectively. To use automatic interval_between_boundaries add '-unsthraut' in the commands:

* 50Salads 
```
python main.py -algo unsup -d 50salads -sp 1 
...
python main.py -algo unsup -d 50salads -sp 5 
```

* GTEA 
```
python main.py -algo unsup -d gtea -sp 1 
...
python main.py -algo unsup -d gtea -sp 4
```

* Breakfast 
```
python main.py -algo unsup -d breakfast -sp 1 
...
python main.py -algo unsup -d breakfast -sp 4
```

## Acknowledgement

 Our code of supervised method is based on a modified versions of [Asformer](https://github.com/ChinaYi/ASFormer) and [MS-TCN](https://github.com/yabufarha/ms-tcn) backbones. We thank the authors of these codebases. 

