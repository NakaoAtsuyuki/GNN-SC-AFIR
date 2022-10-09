# GNN-SC-AFIR

This repository contains the GRRM external codes for GNN/SC-AFIR, GRRM input files, and output EQs performed in the publication "Exploring the quantum chemical energy landscape with GNN-guided artificial force". This external code can enhance the efficiency of the GRRM search by optimizing the direction of the artificial force. Those codes are the external codes for GRRM20.

Citation: 

## External code for GNN/SC-AFIR

### Preparation

Copy codes to the same folder as the GRRM input files. Install the necessary packages according to `conda_requirements.txt`. Python codes need to the execute permission. All codes have to be in the working folder of GRRM. Add the following options in the GRRM input file for using GNN/SC-AFIR.

```
DetailedOutput=ON
SubSelectEQ=./SSE.py
SubPathsGen=./SPG.py
```

Start GRRM calculation as normally after that.

### Output model and transfer learning

The learned GNN model is saved as `[jobname].pth`. 
If you try transfer learning based on a trained model, you prepare a model at a working folder and rename `[jobname].pth`. Change `tf_flag` at line 59 in `Training.py` to True.

### Description of our codes

Our program is divided into 3 systems.

#### Interface of the GRRM program

* `SSE.py`
* `SPG.py`

GRRM call those programs, and those control SC-AFIR search by rewriting GRRM files. When `SSE.py` is called at the first time, it call data reader and machine learning program.

#### Data reader

* `Main.py`
* `Data.py`
* `EQ.py`
* `Task.py`

Those programs run in the background. Those read GRRM output files, and construct a sqlite3 database (`[jobname].db`) for other programs.

#### Machine learning

* `Training.py`
* `Pred.py`
* `MLModel.py`
* `MLTool.py`

`Training.py` and `Pred.py` run in the background. `Training.py` trains GNN model based on a database constructed by `Main.py`. Pred.py evaluates SC-AFIR interventions according to an order by `SSE.py`. 
`MLModel.py` defines model architecture. `MLTool.py` defines input of GNN model. You can change a criterion of success by change `MLTool.py`.

## GRRM inputs

 These are the input files for search in publication "Exploring the quantum chemical energy landscape with GNN-guided artificial force".

* `Glutathione_ML.com` : The input file for search from a glutathione by GNN/SC-AFIR
* `Glutathione_RN.com` : The input file for search from a glutathione by Random/SC-AFIR
* `Tuftsin_ML.com` : The input file for search from a tuftsin by GNN/SC-AFIR
* `Tuftsin_TF.com` : The input file for search from a tuftsin by GNN/SC-AFIR with transfer learning
* `Tuftsin_RN.com` : The input file for search from a tuftsin by Random/SC-AFIR

## List of output EQs

 They are the lists of output EQs file in publication "Exploring the quantum chemical energy landscape with GNN-guided artificial force". Those are xyz format.

* `Glutathione_EQ_list_ML_[1-5].xyz` : The lists of EQs found in the search from glutathione by GNN/SC-AFIR
* `Glutathione_EQ_list_RN_[1-5].xyz` : The lists of EQs found in the search from glutathione by Random/SC-AFIR
* `Tuftsin_EQ_list_ML_[1-5].xyz` : The lists of EQs found in the search from tuftsin by GNN/SC-AFIR
* `Tuftsin_EQ_list_TF_[1-5].xyz` : The lists of EQs found in the search from tuftsin by GNN/SC-AFIR with transfer learning
* `Tuftsin_EQ_list_RN_[1-5].xyz` : The lists of EQs found in the search from tuftsin by Random/SC-AFIR
 

 [1-5] is an index of trial.
