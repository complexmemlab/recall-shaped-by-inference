# recall-shaped-by-inference

[![DOI](https://zenodo.org/badge/808132698.svg)](https://zenodo.org/doi/10.5281/zenodo.12699872)


Repository for the paper "Free recall is shaped by inference and scaffolded by event structure" 

Organization:
```yaml
├── src : package containing the python code used for the papers analyses
│   ├── data : code for processing the data and generating demographics information
│   └── models : code for the reinforcement learning model
├── data : all data analyzed in the paper
│   ├── raw : raw data files (these have been moved to the OSF https://osf.io/3d746/
│   │   ├── exp1: Main manuscript files
│   │   └── exp2: Replication files
│   ├── interim : data that has undergone some processing
│   │   ├── exp1: Main manuscript files
│   │   └── exp2: Replication files
│   └── processed : processed data
│   │   ├── exp1: Main manuscript files
│   │   └── exp2: Replication files
└── results : all results generated in the paper
    └── figs : png and pdf copies of data figures used in paper
```

## One time setup
Here we describe two options for recreating our computational environment using Conda. 

### Conda
1. Install the anaconda python distribution on your computer using appropriate guide below (I would recommend the command line utility):
    - [OSX](https://docs.anaconda.com/anaconda/install/mac-os/)
    - [Windows](https://docs.anaconda.com/anaconda/install/windows/)
    - [Linux](https://docs.anaconda.com/anaconda/install/linux/)
2. Once anaconda is installed run `conda init` in the terminal
3. Navigate to the repository on your computer and run `conda create -n writ-fr --file environment.yml`
4. Once the environment is created run `conda activate writ-fr` 
5. Then you must run `pip install -e .` 
6. Finally you can run `jupyter notebook`
7. Launch any given notebook under scripts/notebooks folder: (01-data-cleaning.ipynb, 02-paper-notebook.ipynb, 03-supplement.ipynb)

Please feel free to send me an email at a.b.karagoz@wustl.edu or post an issue if you are having difficulties running any component of this code.
