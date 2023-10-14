=======================================================================
TITLE: Classification Model for Airline Travel Experience. 
=======================================================================

DESCRIPTION :   The project is about building a MLops process. 
				A classification model is trained on Airline Travel experience of passengers
				Model is built using DecisionTree Classifier from scikit-learn.
				A basic model which is able to predict whether a passenger is satisfied or dissatisfied based on several predictors

++++++++++++++++++++++++++++++++++++++++++++++
INSTALLATION: 
++++++++++++++++++++++++++++++++++++++++++++++
Cloned Repository:
	1) The cloned repository has a virtual environment already set with required dependencies to build and run the model.
	2) To activate the environment,
		In PowerShell: <virual_env_folder>/Scripts/activate.ps1
		In Linux/Unix-Based: <virtual_env_folder>/Scripts/activate
		In Windows CMD: <virtual_env_folder>/Scripts/activate.bat

For running on a host machine:
Windows: 
	Python:
	1) Visit the https://www.python.org/downloads/release/python-391/
	2) Run the installer and follow the instructions
	3) Add the python 3.9 to "PATH" env variable.
	4) Verify installation by running "python --version" in the CMD.
	
	Python libraries: 
		scikit-learn:
		1) run "pip install -U scikit-learn==1.3.1" to install
		2) verify using "python -m pip show scikit-learn"
		
		pandas:
		1) run "pip install pandas==2.1.1"
		2) verify using "python -m pandas --version"
		
		numpy:
		1) run "pip install numpy==1.26.0"
		2) verify using "python -m numpy --version"
		
		venv:
		1) venv comes with python 3.9.1 as standard library.
		2) verify using "python -m venv --version"

Linux/Unix-Based:
	Python:
	1) check if python 3.9.1 is already installed.
	2) if not then run 
			"sudo apt update"
			"sudo apt install python3.9"
	3) verify using python3 --version
	
	Python libraries:
		scikit-learn:
		1) run "pip install -U scikit-learn==1.3.1" to install
		2) verify using 'python3.9 -c "import sklearn; print(sklearn.__version__)"'
		
		pandas:
		1) run "pip install pandas==2.1.1"
		2) verify using "python3.9 -c "import pandas; print(pandas.__version__)""
		
		numpy:
		1) run "pip install numpy==1.26.0"
		2) verify using "python3.9 -c "import numpy; print(numpy.__version__)""
		
		venv:
		1) venv comes with python 3.9.1 as standard library.
		2) verify using "python3.9 -c "import venv; print(venv.__version__)""


-------------------------------------------------------------
USAGE: 
-------------------------------------------------------------
The model will be trained and accessed via class "AirLine_ML".
All the required methods are inside this class.
The model itself is very basic and can be improved by pruning and Hyper-Parameters.
For now, It servers the purpose of ML model for our MLops process.

Following is a code snippet explaining how to access and run.

>>>from classModel import AirLine_ML
>>>if __name__ == "__main__":
		path = "datasets\Airline_Satisfaction.csv"
		
		# create instance of class
		model = Airline_ML()
		
		# get the classifier model and the values
		clf, cat_features, y_mapped, x_mapped = model.train_model()
		
		# To get sample input
		main_df = pd.read_csv(path, index_col=0)
		#print(main_df['Type of Travel'].value_counts())
		#print(main_df['Customer Type'].value_counts())
		#print(main_df['Class'].value_counts())
    
		# select all the columns except first and last column as they are not required
		input_row = main_df.iloc[:1,1:-1].copy(deep =True)
		expected = main_df.iloc[:1,-1:].copy(deep = True)
    
		print("Input row before mapping: ",input_row)
		print("Target mapped : ",y_mapped)
		print("categories mapped : ", x_mapped)
    
		# mapping the corresponding values.
		input_row = model.mapper(input_row, x_mapped)
		print("Input row after mapping: ",input_row)
  
  
		print("Expected : ", y_mapped[expected['satisfaction'][0]])
		print("predicted : ", clf.predict(input_row)[0])

=============================================================		
DATA: 
=============================================================
In the cloned repository:	
	The dataset is in the following path, "datasets\Airline_Satisfaction.csv"

From GitHub repository:
	Follow Link: 
	


	
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
TRAINING: 
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
The data was preprocessed and the loaded into the dataframe as this model is for MLops
The model was trained by splitting the dataset into Train-Test dataframes
Then a classifier object was created and then fitted for train test dataframes





|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
RESULTS:
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Classification Report
:                precision    recall  f1-score   support

           0       0.91      0.93      0.92      3382
           1       0.94      0.93      0.94      4411

    accuracy                           0.93      7793
   macro avg       0.93      0.93      0.93      7793
weighted avg       0.93      0.93      0.93      7793

Accuracy 	:  0.9294238419094059
AUC 		:  0.9448794483336307





```````````````````````````````````````````````````````````````
CONTRIBUTING: 
```````````````````````````````````````````````````````````````
Anyone can clone the repository for their use and can contribute as this is for MLops process.
The classifier is a basic Decision Tree model that can be improved in various ways, So feel free to contribute.



:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
ACKNOWLEDGMENTS: 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Credits:
	Scikit Learn 	: https://scikit-learn.org/
	pandas 			: https://pandas.pydata.org/
	numpy			: https://numpy.org/

###############################################################
CONTACT: 
###############################################################
Email ID	: dixit.b.c@gmail.com
GitHub  	: www.github.com/iCODEalien
LinkedIn	: www.linkedin.com/in/dixitbc

