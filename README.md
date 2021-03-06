[Kaggle](www.kaggle.com) competition to predict dog outcomes for the Austin Animal Center.

### Data Files

| File Name | Available Formats |
|-----------|:-----------------:|
|sample_submission.csv  |  .gz (15.10 kb)|
|test.csv  | .gz (190.70 kb)|
|train.csv |  .gz (521.35 kb)|

The data comes from [Austin Animal Center](http://www.austintexas.gov/department/animal-services) from October 1st, 2013 to March, 2016. Outcomes represent the status of animals as they leave the Animal Center. All animals receive a unique Animal ID during intake. 

In this competition, you are going to predict the outcome of the animal as they leave the Animal Center. These outcomes include: Adoption, Died, Euthanasia, Return to owner, and Transfer. 

The train and test data are randomly split. 

### File descriptions

- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Setup

- To set up your dev env, [follow these instructions](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#anaconda-installation).
- If you have issues installing tensorflow via conda, [follow these instructions](http://vinhdq.blogspot.com/2015/12/installing-tensorflow-on-mac-os-1011.html).
- [Run a quick test](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#test-the-tensorflow-installation) to make sure tensorflow has been properly installed.
- To activate the python virtualenv:
	> `source activate tensorflow`
- Install jupyter notebook:
	> `pip install jupyter`
- Once you're done, deactivate the virtualenv:
	> `source deactivate`
