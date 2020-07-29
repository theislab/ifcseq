# IFC-Seq: predicting gene expression for Imaging Flow Cytometry data

![logo](fig1.png)

**Description**

With IFC-Seq we couple two distinct modalities of single-cell data: **(1) Single Cell Transcriptomics (SCT)** and **(2) Imaging Flow Cytometry (IFC)**.  Both modalities are aligned using common surface markers (CD34 and FcgR in this example). By leveraging both views at the same time, IFC-Seq is able to predict gene expression at the single-cell level for data acquired from IFC experiments.

This is a supplementary tutorial to the methodology described in [N.K. Chlis et al., 2019](https://www.helmholtz-muenchen.de/icb/index.html). A detailed step-by-step tutorial is presented in tutorial_ifcseq_mouse.ipynb. A concise demnostration of how to use ifcseq as a python package is demonstrated in tutorial_ifcseq_human.ipynb.

**Instructions**
1. Clone the repository by using "git clone https://github.com/theislab/ifcseq.git"
2. Download the mouse data from [here](https://drive.google.com/file/d/1QGwe6f_gKyCJHwhttbrvWSp9T_tzchc0/view?usp=sharing) and unzip into the repository, it should automatically unzip into ./ifcseq/ifcseq_mouse_data/
3. Download the human data from [here](https://drive.google.com/file/d/1WkrLAPka43rMqCyuuieMFuryLEGjfGYI/view?usp=sharing) and unzip into the repository, it should automatically unzip into ./ifcseq/ifcseq_human_data/
4. Open and run the tutorial notebook of choice

**List of materials**
1. [tutorial_ifcseq_mouse.ipynb](./tutorial_ifcseq_mouse.ipynb): The notebook presenting IFC-Seq in detail, step by step on the mouse data
1. [tutorial_ifcseq_human.ipynb](./tutorial_ifcseq_human.ipynb): The notebook demonstrating how to use the ifcseq pyhon package on the human data. Using ifcseq as a python package assumes that the surface markers are available for the IFC dataset. Either expermentally, or predicted in a label-free manner as shown in the previous notebook on the mouse data.
3. [CNN_train.py](./CNN_train.py): Python code used to train the CNN keras model which is used in tutorial_ifcseq_mouse.ipynb. For convenience, a pre-trained model is availabe in ifcseq_mouse_data.zip (see below)
4. [ifcseq_mouse_data.zip](https://drive.google.com/file/d/1QGwe6f_gKyCJHwhttbrvWSp9T_tzchc0/view?usp=sharing): The mouse data and pre-trained models necessary to run the notebook
5. [ifcseq_human_data.zip](https://drive.google.com/file/d/1WkrLAPka43rMqCyuuieMFuryLEGjfGYI/view?usp=sharing): Additional human data used in the publication.
