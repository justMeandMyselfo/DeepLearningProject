# DeepLearningProject
#This section is mostly for my self but if any one finds it feel free to read it.

The idea to this project was to build a deep learning model to recognize 101 different foods and estimate the calories of it. 
Dataset food101N which is food101 with added noises. I would recommand to add an extra folder of random images so that if you encounter an unknown food then it doesn't say any random food. + if I have the time I would add a way to enter the food's name if it's called unknown so that the user can also help training the model.

Two different models were built, first one has a lower accuracy but is much lighter and faster to train, and so works perfectly fine on a smartphone thanks to TFlite.->Model for mobile.

Second one is more accurate, slower to train (need GPU to train) and because of the functions added it is not possible to use it on a phone, but we could host that model on a server. -> HighAccuracyModel.
