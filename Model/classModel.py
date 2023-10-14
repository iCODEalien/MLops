import numpy as np
import pandas as pd
#import os
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, classification_report, confusion_matrix

class Airline_ML():
    def __init__(self):
        self._path = "datasets\Airline_Satisfaction.csv"
        self.target_name = 'satisfaction'
        
    # Extract Numerical and Categorical features.
    def extract_num_cat(self,df):
        # Pulling out numerical features by conditioning dtypes not equal to object type
        numerical_features = df.dtypes[df.dtypes != "object"].index
        
        # Pulling out categorical features 
        categorical_features = df.dtypes[df.dtypes == "object"].index
        
        return numerical_features, categorical_features
    
    # encoding.
    def encoder(self,df,categorical_features = [], target = False):
        if target:
            unq_values = list(df.unique())
            if len(unq_values) == 2:
                if all(str(x).isnumeric() for x in unq_values):
                    print("All values are numeric, Hence encoded.")
                    return df
                else:
                    df = df.replace([unq_values[0],unq_values[1]], [0,1])
                    #mapped dict
                    mapped_dict = {}
                    for keys,value in zip(unq_values,[0,1]):
                        mapped_dict[keys] = value
                        
                    return df,mapped_dict
                
            elif len(unq_values) > 2:
                if all(str(x).isnumeric() for x in unq_values):
                    print("All values are numeric, Hence encoded.")
                    return df
                else:
                    mapped_dict = {}
                    num_list = list(range(len(unq_values)))
                    for key,value in zip(unq_values, num_list):
                        df = df.replace(key,value)
                        mapped_dict[key] = value
                    return df, mapped_dict
            else:
                print("Error, target has only one value")
                
        else:
            mapped_dict = {}
            if len(categorical_features) > 0:
                for cat in categorical_features:
                    value_code_tuple = []
                    values = list(df[cat].unique())
                    if len(values) == 2:
                        df[cat] = df[cat].replace([values[0],values[1]], [0,1])
                        value_code_tuple.append((values[0],0))
                        value_code_tuple.append((values[1],1))
                    else:
                        #encoded_1h = pd.get_dummies(df[cat],prefix = cat, drop_first=True)
                        #df = pd.concat([df,encoded_1h], axis = 1)
                        #df.drop(cat, axis = 1, inplace=True)
                        for i in range(len(values)):
                            df[cat] = df[cat].replace(values[i],i)
                            value_code_tuple.append((values[i],i))
                    mapped_dict[cat] = value_code_tuple
                return df, mapped_dict
    
    # Treating the outliers
    def treat_outliers(self,df, numerical_features):
        for i in numerical_features:
            if i != "Numcolumn":
                #print("-"*90)
                #print(i)
                IQR = df[i].quantile(0.75) - df[i].quantile(0.25)
                UL = df[i].quantile(0.75) + 1.5*IQR
                LL = df[i].quantile(0.25) - 1.5*IQR
                #print("IQR:",IQR)
                #print("UL: ",UL)
                #print("LL:",LL)
                df[i] = np.where(df[i]>UL, UL,df[i])
                df[i] = np.where(df[i]<LL, LL, df[i])
                outliers = len(df[(df[i] < LL) | (df[i] > UL)])
                #print("outliers: ",outliers)
        return df
    
    def mapper(self, input_df, mapping_dictionary):
        for col, mapping_list in mapping_dictionary.items():
            category_to_integer = dict(mapping_list)
            input_df[col] = input_df[col].map(category_to_integer)
        return input_df
        
    def train_model(self):
        # storing and reading the csv file using pandas into a pandas dataframe
        df = pd.read_csv(self._path, index_col=0)
        #print(df.head())
        
        # dropping the id column    
        df.drop('id',axis = 1, inplace = True)
        
        # Predictors
        df_x = df.drop(self.target_name, axis = 1)
        
        # target
        df_y = df[self.target_name].copy(deep = True)
        # numerical attribtes and categorical attributes.
        num_attr , cat_attr = self.extract_num_cat(df_x)
        
        # Treating Outliers.
        df_x = self.treat_outliers(df_x, num_attr)

        # encoding
        df_x, x_mapped = self.encoder(df_x,cat_attr)
        df_y, y_mapped = self.encoder(df_y, target = True)
        
        # Train and Test data
        X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.3, random_state = 42)
        
        # Classifier Model
        clf = DecisionTreeClassifier()
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        #print(y_pred)
        
        # crosstab
        CrossTab = pd.crosstab(y_test, y_pred, margins = True, rownames=['Actual'], colnames=['Predicted'], margins_name="All")
        #print("Cross Tabulation : \n",CrossTab)
        print("Classification Report \n: ",classification_report(y_test,y_pred))
        print("Accuracy : \n",accuracy_score(y_test, y_pred))
        
        # FPR, TPR, AUC
        y_pred_prb = clf.predict_proba(X_test)[:,-1]
        #print("y_pred_prob shape : ",y_pred_prb.shape)
        
        #fpr, tpr, thresholds = roc_curve(y_test,y_pred_prb)
        #print("fpr: ",fpr[0:5])
        
        auc = roc_auc_score(y_test,y_pred_prb)
        print("AUC : ",auc)

        # returning trained classifier, categorical_features, target values mapped dict.
        return clf, cat_attr, y_mapped, x_mapped
        

#script
if __name__ == "__main__":
    path = "datasets\Airline_Satisfaction.csv"
    
    model = Airline_ML()
    
    clf, cat_features, y_mapped, x_mapped = model.train_model()
    
    main_df = pd.read_csv(path, index_col=0)
    #print(main_df['Type of Travel'].value_counts())
    #print(main_df['Customer Type'].value_counts())
    #print(main_df['Class'].value_counts())
    
    # select all the columns except first and last column as they are not required
    input_row = main_df.iloc[:1,1:-1].copy(deep =True)
    expected = main_df.iloc[:1,-1:].copy(deep = True)
    
    print("Input row before: ",input_row)
    print("Target mapped : ",y_mapped)
    print("categories mapped : ", x_mapped)
    
    # mapping the corresponding values.
    input_row = model.mapper(input_row, x_mapped)
    print("Input row after: ",input_row)
    
    #main_df = model.encoder(main_df.iloc[:1,1:-1], categorical_features=cat_features)
    #print(main_df.head())
    #print(main_df.info())
    
    print("Expected : ", y_mapped[expected['satisfaction'][0]])
    print("predicted : ", clf.predict(input_row)[0])
