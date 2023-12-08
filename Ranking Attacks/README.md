## Running the Code

We run the code using Visual Studio Code as it provides tools that makes the experience easier.

1. Ensure that you have basic python packages installed (scikit-learn,numpy).
2. The main file is (`Ranking_of_noises.ipynb`)
3. To use it, you need to put outputs of your models in the (`/datasets/`) folder, under (`/datasets/base/`) if it is the ouput on base CIFAR-10 dataset, and under (`/datasets/noisy/<attack-name>`) if it is the ouput on CIFAR-10 dataset attacked using *attack-name*
4. The output files should use the name of the model, which we plan on using in `Ranking_of_noises.ipynb`
5. `Ranking_of_noises.ipynb` has a few sections: First to collect accuracy readings from the output files, then, to rank the attacks.
6. The two sections need to be executed serially, and the results can be seen in `/results/`
