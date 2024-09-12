
# phys-467-exercises -- fall semester 2024

Repository for the exercise sessions for the phys-467 class at EPFL, fall semester 2024.

## How to generate links to open the notebooks in noto@epfl

[Noto](https://www.epfl.ch/education/educational-initiatives/jupyter-notebooks-for-education/one-click-access-to-jupyter-notebooks-online-with-noto/) is EPFL's version of Google colab.
It allows to edit and run jupyter notebooks easily. 

To generate a link that opens noto, automatically clones (or pulls) this repo, and opens a given notebook, you can use this [link generator](https://nbgitpuller.readthedocs.io/en/latest/link.html), with parameters

    JupterHub URL: https://noto.epfl.ch
    Git repository URL: https://github.com/SPOC-group/phys-231-exercises
    Branch: main
    File to open: path to the notebook that must be opened, relative to the root of the git directory. For example "lecture1/file_name.ipynb"
    Application to open: JupyterLab
    Named Server to open: blank

or use this link, substituting only the path parameter

    https://noto.epfl.ch/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FSPOC-group%2Fphys-231-exercises&urlpath=lab%2Ftree%2Fphys-231-exercises%2FExercise_3%2Fex_3_images_and_svd.ipynb&branch=main

Then, [GO EPFL](https://go.epfl.ch) can be used to generate a shorter version of the link.
