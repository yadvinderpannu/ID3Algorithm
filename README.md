# ID3Algorithm
ID3 Algorithm for generating decision tree used to predict class of Iris plant.
The dataset is taken from the UCI Machine Learning Repository. It can be found in this link:
https://archive.ics.uci.edu/ml/datasets/iris

1. The program takes two arguments as an input.
* The first argument is the attribute filename. 
* The second argument is the dataset filename.

For example, if code file is ID3Algorithm.py and 
			attribute file is attribute.txt
			and dataset file is dataset.txt
      and you are trying to execute the program from the command prompt/terminal type the following:
	    python ID3Algorithm.py attribute.txt dataset.txt
      
After the program begins execution, it will ask which type of implementation you want.
You can select from data split or K fold implementation by specifying 1 or 2.
You can select any. If you select K fold implementation, it will ask for the value of k. Enter it there.


2. For the program the following functions were created:
* split_data function to split data into 80% training and 20% testing
* k_fold_split function for k fold implementation, user specifies the number of folds at the runtime. 
       Depending upon the number of folds, the training and testing data will be manipulated.
* get_best_attribute function for determining the best attribute by entropy calculation and information gain.
Conventional formulas are applied for calculating the entropy and the information gain.
		   Checks which attribute best splits the data.
* dtree_train function to recursively determine nodes of d tree
* predict function to predict the class for the test data using the obtained d tree
* print_tree function to finally display the tree 
* Execution starts from the main function where it is determined whether to do data split or k fold based on user input.


3. The prediction accuracy and the printed out tree are shown in multiple screenshots.

