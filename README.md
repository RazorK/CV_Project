# CV_Project
A project to predict evaluations of stores by giving only pictures.

## Setup:

+ Install mysql-server, mysql-client(input and remember the password):

  `apt-get install mysql-server mysql-client`

+ Install mysql-python through anaconda:

  `conda install mysql-python`

+ import the yelp database(put the .sql file in the current path):

  `mysql -uroot -p`

  `create database yelp_db;`

  `use yelp_db`

  `source pho_d.sql`

  `source photo.sql`

  `source bus_d.sql`

  `source business.sql`

+ create a local example configure file:

  `cp configure_example.py configure.py`

  `sudo vim configure.py`

  and then change the local parameters


## File Structure

#### configure_example.py

the example file for configure.py. 

#### configure.py

configure file.

#### dataset.py

dataset class. 

Class YelpDataSet Inherited from pytorch dataset:

Parameters:

- photo_dir: the photo directory for the yelp dataset.
- category: the category for the photo.
  - 'food', 'inside', 'outside', 'menu', 'drink'
- transform: transform functions for image.

#### dataLoader.py

data loader, provide a function to randomly pick train set and validation set:

##### get_train_valid_loader:

Parameters:

+ data_dir: the photo_dir
+ batch_size
+ random_seed: int or list of int
+ category: the category for the photo.
  + 'food', 'inside', 'outside', 'menu', 'drink'
+ transform: transform function, default None
+ valid_size: float, 0~1, the radio of valid_size, default 0.1
+ shuffle: default True

#### CV_Project.ipynb

The model design and train period.

