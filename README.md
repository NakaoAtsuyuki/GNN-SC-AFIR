# GNN-SC-AFIR

This repository contains GRRM external code, GRRM input and output EQs performed in publication "Exploring the quantum chemical energy landscape with GNN-guided artificial force"

Citation: 

## External code for GNN/SC-AFIR
Those codes are external codes for GRRM20.

### Preparation
Copy codes at folder GRRM input files. Install the necessary packages according to conda_requirements.txt file. The all codes has to be at working folder of GRRM. Use the following options in a GRRM input file for using GNN/SC-AFIR.

```
DetailedOutput=ON
SubSelectEQ=./SSE.py
SubPathsGen=./SPG.py
```

Start GRRM calculation as normally after that

### Output model and transfer learning

The learned GNN model is saved as [jobname].pth. 

If you try transfer learning based on a trained model, you prepare a model at working folder and rename [jobname].pth. Change tf_flag at line 59 in Training.py to True.

### Description of our codes

Our program is divied to 3 system.

#### Interface to GRRM program

* SSE.py
* SPG.py

GRRMs call those program and those control SC-AFIR search by rewriting GRRM files. When SSE.py is called first, SSE.py start bellow programs.

#### Data collecter

* Main.py
* Data.py
* EQ.py
* Task.py

Those programs run in background. Those read GRRM output files, and construct sqlite3 database ([jobname.db]) for other programs.

#### Machine learning

* Training.py
* Pred.py
* MLModel.py
* MLTool.py

Training.py and Pred.py run in backgroud. Training.py trains GNN model based on database constructed by Main.py. Pred.py evaluate SC-AFIR interventions according to the order by SSE.py. 

MLModel.py defines model architecture. MLTool.py defines input of GNN model. You can change a craiterion of succsess by change MLTool.py

## GRRM inputs

 They are the input files for search in publication "Exploring the quantum chemical energy landscape with GNN-guided artificial force".
 
 * Glutathione_ML.com : Input file for search from a glutathione by GNN/SC-AFIR
 * Glutathione_RN.com : Input file for search from a glutathione by Random/SC-AFIR
 * Tuftsin_ML.com : Input file for search from a tuftsin by GNN/SC-AFIR
 * Tuftsin_TF.com : Input file for search from a tuftsin by GNN/SC-AFIR with transfer learning
 * Tuftsin_RN.com : Input file for search from a tuftsin by Random/SC-AFIR

## List of output EQs

 They are the lists of output EQs file in publication "Exploring the quantum chemical energy landscape with GNN-guided artificial force". Those are xyz format.
 
  * Glutathione_EQ_list_ML_[1-5].com : the lists of EQs found in the search from glutathione by GNN/SC-AFIR
  * Glutathione_EQ_list_RN_[1-5].com : the lists of EQs found in the search from glutathione by Random/SC-AFIR
  * Tuftsin_EQ_list_ML_[1-5].com : the lists of EQs found in the search from tuftsin by GNN/SC-AFIR
  * Tuftsin_EQ_list_TF_[1-5].com : the lists of EQs found in the search from tuftsin by GNN/SC-AFIR with transfer learning
  * Tuftsin_EQ_list_RN_[1-5].com : the lists of EQs found in the search from tuftsin by Random/SC-AFIR
  
 [1-5] is an index of trial.
