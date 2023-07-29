# Bayesian Hyperparameter Tunning

Use the Bayesian optimisation algorythm to tune simple TF deep learning model (For fast experimentation)

## Versionnig of the used packages
- Tensorflow 2.12
- Numpy 1.23.5
- GPy 1.10.0
- GPyOpt 1.2.6

## How to use it
Find in the `parameter_space.py` file the bounds values for the deep learning model hyperparameter. In the file exemple, we tune
- Learning rate
- Beta 1 & 2 from the Adan optimizer
- Layers nodes units
- Batch size
- Dropout values for all dropout layers

The, with the exemple model build and train function, run the main function, it will optimise an hand writen classical deep learning model classification
```
python hyperparameter_tuning.py
```

## Result
On the model, got a solid 98% of accuracy by tunned the model. See result on the .txt outputfile in the repository

## License
No license, you can use it 

---
Work done in extend to a existing [school project](https://github.com/Camaltra/holbertonschool-machine_learning/tree/main/unsupervised_learning/hyperparameter_tuning).