# NESA Software Engineering - Machine Learning OOP Implementation Examples

This Jupyter Notebook collection is designed to support students to implement [Programming for automation](https://curriculum.nsw.edu.au/learning-areas/tas/software-engineering-11-12-2022/content/year-12/fa56cc30c8#cg-1f80854c-5ce4-4d68-aa18-9a6fc0861c8c) in the [NESA Software Engineering Syllabus](https://curriculum.nsw.edu.au/learning-areas/tas/software-engineering-11-12-2022/overview) specifically using an OOP approach to make predictions. Open these Jupyter Notebooks in [Jupyter Notebook](https://jupyter.org/try-jupyter/lab/), VSCode or Codespaces to modify the code/data and run the code blocks.

> [!Important]
> The configuration for VSCode and Codespaces has been built into this repository. To ensure the necessary dependencies are installed and configured it is recommended that a GitHub Codespace is used for the Neural Network implementations (notebooks 6, 7 and 8), which need TensorFlow.
>
> If you run the notebooks locally instead, install the Python packages with `pip install -r requirements.txt`. The tree diagram at the end of notebook 9 also needs the [Graphviz](https://graphviz.org/download/) program installed and on your PATH (the Codespace installs it automatically).

---

## Demonstrations

1. [OOP Linear Regression Implementation](examples/1.OOP_Single_Feature_Linear_Regression.ipynb).
2. [OOP Multiple Feature Linear Regression Implementation](examples/2.OOP_Multi_Feature_Linear_Regression.ipynb).
3. [OOP Polynomial Linear Regression Implementation](examples/3.OOP_Polynomial_Liner_Regression.ipynb).
4. [OOP Logistic Regression Classification Implementation](examples/4.OOP_Logistic_Regression.ipynb).
5. [OOP K-Nearest Neighbour Classification Implementation](examples/5.OOP_KNN.ipynb).
6. [OOP Neural Network Linear Regression Implementation](examples/6.OOP_Neural_Network_Simple.ipynb) - includes notes aligning the implementation to Software Engineering Course Specifications page 29 on Neural Networks.
7. [OOP with some MLOps Neural Network Intermediate Classification Implementation](examples/7.OOP_MLOps_Neural_Network_Intermediate.ipynb).
8. [OOP Neural Network Image Classification Implementation](examples/8.OOP_Neural_Network_Adv.ipynb).
9. [OOP Decision Tree Classification Implementation](examples/9.OPP_Decision_Trees.ipynb).
10. [OOP K-Means Clustering Implementation](examples/10.OOP_K_Means_Clustering.ipynb) - an unsupervised learning example that groups iris flowers without using their species, including the elbow method for choosing k and a step-by-step view of the algorithm.

The `examples/py_scripts` folder contains the Course Specifications linear regression examples as standalone Python scripts. They can be run from any folder.

## Licence

This collection is adapted from [Machine Learning OOP Implementation Examples](https://github.com/TempeHS/Machine_Learning_OOP_Implementation_Examples) by [Ben Jones](https://github.com/benpaddlejones), used under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Changes by [Ryan Dunne](https://github.com/DunneTeaches): edited the notebooks and added notebook 10 (K-Means Clustering).

This adapted collection is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) by [Ryan Dunne](https://github.com/DunneTeaches).

Original licence notice:
<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/TempeHS/Machine_Learning_OOP_Implementation_Examples">Machine Learning OOP Implementation Examples
</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://github.com/benpaddlejones">Ben Jones</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block; ">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International<img style="height:22px!important; margin-left:3px; vertical-align:text-bottom; " src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important; margin-left:3px; vertical-align:text-bottom; " src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important; margin-left:3px; vertical-align:text-bottom; " src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important; margin-left:3px; vertical-align:text-bottom; " src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>
