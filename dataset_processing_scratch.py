import numpy as np
import pandas as pd
import math
import logging
from pandas.api.types import is_numeric_dtype
logging.basicConfig(level=logging.ERROR)


class DatasetProcess:
    def __init__(self, file_fullname, target_column,data_ratio):
        self.dataset = pd.read_csv(file_fullname)
        self.data_ratio = data_ratio 
        self.target_column = target_column
        self.column_names = None
        self.dataset_dummies = None
        self.training_set = None
        self.testing_set = None
        self.training_stats = None
        self.scale_columns = None

        if target_column not in self.dataset.columns:
            logging.error(f"{target_column} was not found in the dataset")
            raise DatasetError(
                f"{target_column} does not exist"
            )
        if not (0 < data_ratio < 1):
            logging.error(f"Training ratio must be above zero and less than 1")
            raise DatasetError(
                    f"{data_ratio} must be between 0.0 t0 1.0"
            ) 
        
        self.prepare()
        
        
    def prepare(self): 
        """ split_data function do not get called when data is not split """
    
        self.column_names, self.dataset_dummies  =  self.category_to_binary()
        self.training_set, self.testing_set = self.split_data()
        self.training_stats,self.scale_columns = self.prepare_scaling_data()
    
    def category_to_binary(self):
        """Converts neccessary columns to bits (0 & 1)"""
        
        column_names = self.dataset.columns
        dataset_dummies = self.dataset.copy()
        
        for i in column_names:
             
             if len(self.dataset[i].unique()) > 2 and not pd.api.types.is_numeric_dtype(self.dataset[i]):
                 dataset_dummies  = pd.get_dummies(dataset_dummies , columns=[i],dtype=int)
             
        column_names = dataset_dummies.columns

        for i in column_names:
            if len(dataset_dummies[i].unique()) == 2 and not pd.api.types.is_numeric_dtype(dataset_dummies[i]):
                first = (dataset_dummies[i].unique()).tolist()[0]
                second = (dataset_dummies[i].unique()).tolist()[1]
                dataset_dummies[i] = dataset_dummies[i].map({first:1,second:0})
                
        return column_names, dataset_dummies

    def split_data(self):
        dataset_shuffled = self.dataset_dummies.sample(frac=1,random_state=42).reset_index(drop=True)
        data_length = len( self.dataset_dummies)
        split_value = int(self.data_ratio * data_length )
        training_set = dataset_shuffled[:split_value ]
        testing_set = dataset_shuffled[split_value :]
        
        return training_set, testing_set 

    def prepare_scaling_data(self):
        
        training_stats = {}  
        scale_columns = []

        for col in self.column_names:
            if is_numeric_dtype(self.dataset_dummies[col]):
                scale_columns.append(col)

        for i in scale_columns:
            training_stats[i]={'min': self.training_set[i].min(),'max': self.training_set[i].max()}

        return training_stats,scale_columns


    def transform_columns(self):
        """This scales all the numeric columns"""
        
        training_set_scale = self.training_set.copy()
        testing_set_scale = self.testing_set.copy()
        
        for each_column in self.scale_columns:
            max_value = self.training_stats[each_column]['max']
            min_value = self.training_stats[each_column]['min']

            if max_value == min_value:
                logging.error(f"The {each_column} values are same")
                raise DatasetError(
                    f"The {each_column} values are same."
                )
            
            training_set_scale[each_column ] = ((training_set_scale[each_column ] - min_value)/ (max_value - min_value))
            testing_set_scale[each_column ] = ((testing_set_scale[each_column ] - min_value)/ (max_value - min_value))

        X_train = training_set_scale.drop(columns=[self.target_column]).values
        y_train = training_set_scale[self.target_column].values
        X_test = testing_set_scale.drop(columns=[self.target_column]).values
        y_test = testing_set_scale[self.target_column].values
       

        return  X_train, y_train , X_test, y_test,self.dataset_dummies.columns 
        

class DatasetError(Exception):
    """ Base class for errors """
    pass
