import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numba import jit
from scipy.signal import argrelextrema  
import itertools
import statistics
#from Data_generation import GeneratedData

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'ScaledCoordinates_Post-Trial.xlsx')
DICTIONARY_PATH = os.path.join(os.path.dirname(__file__), 'Dictionary_Kinematics.xlsx')


class CowsDataset:
    def __init__(self, sheet_names=None):
        self.sheet_names = sheet_names
        self.df = self.load_dataset()
        self.i = 0


    def __getitem__(self, index):
        return self.df[index]

    def __len__(self):
        return len(self.df)

    def __next__(self):
        if self.i < len(self):
            self.i += 1
            return self[self.i-1]
        raise StopIteration()

    def __iter__(self):
        return self

    def load_dataset(self):
        """

        :return: load the given dataset
        """
        dictionary = pd.read_excel(DICTIONARY_PATH, "Video File -> Excel Tab Names")
        dictionary = dictionary.to_numpy()
        if self.sheet_names is None:
            self.sheet_names = dictionary[:, 0]
        df = [pd.DataFrame(pd.read_excel(DATASET_PATH, sheet).iloc[:, 0:34]) for
              x, sheet in enumerate(self.sheet_names)]
        return df
    
    
    def remove_steps_columns(self):
        """
        

        Returns dataframe without the first 2 columns that indicate the begin of 
        front and back step
        -------
        None.

        """
        df = [df.iloc[:, 2:34] for i, df in enumerate(self.df)]  
        
        return df
              
    
   
    def get_head(self, n_rows):
        """
        

        Returns dataframe with the values of n_rows first rows
        -------
        None.

        """
        df = [df.head(n_rows).values for i, df in enumerate(self.df)]
        return df
    
    @staticmethod
    def get_side_sheets(side):
        """

        :param side: side or side2 as string
        :return: the names of the excel sheet of the given side
        """
        dictionary = pd.read_excel(DICTIONARY_PATH, "Video File -> Excel Tab Names")
        dictionary = dictionary.to_numpy()
        if side == 'side1':
            side1_names = dictionary[0::2]
            return side1_names[:, 0]
        elif side == 'side2':
            side2_names = dictionary[1::2]
            return side2_names[:, 0]

    @staticmethod
    def get_sheet(sheet_name):
        """

        :param sheet_name: sheet name (string)
        :return: the excel sheet corresponding the given sheet name
        """
        return pd.DataFrame(pd.read_excel(DATASET_PATH, sheet_name).iloc[:, 0:34])

    @staticmethod
    def get_cow_names():
        """

        :return: a list of the cows' names
        """
        dictionary = pd.read_excel(DICTIONARY_PATH, "Video File -> Excel Tab Names")
        dictionary = dictionary.to_numpy()
        side1_names = dictionary[0::2]
        return side1_names[:, 1]

    @staticmethod
    def get_cow_name(sheet_name):
        """

        :param sheet_name: sheet's name (string)
        :return: the name of the cow corresponding the name of the sheet
        """
        dictionary = pd.read_excel(DICTIONARY_PATH, "Video File -> Excel Tab Names")
        dictionary = dictionary.to_numpy()
        index = np.where(dictionary == sheet_name)
        return dictionary[index[0], 1][0]
    
    @staticmethod
    def get_joint_names(dataset):
        """

        :return: list of the joint names
        """
        joint_names = dataset[1].iloc[0, 0:34].index
        return joint_names

    def get_list_of_joints(self):
        """

        :return: list of the joints
        """
        joint_names = CowsDataset.get_joint_names(self)
        list_column = []
        for i, column in enumerate(joint_names):
            for j, dCow in enumerate(self.df):
                list_column.append(dCow[column])  # list containing all the columns from all the sheets/columns with the same column_name are together
        df_column = pd.DataFrame(list_column).T  # Dataframe containing the same column name of diffrent sheet next to each other
        list_of_joints = []
        for i, column in enumerate(joint_names):
            df_joint = df_column.filter(regex=column)
            df_joint.columns = CowsDataset.get_cow_names() + "   " + df_joint.columns
            list_of_joints.append(df_joint)
        return list_of_joints
    
    @staticmethod
    def get_extrema_indexes(data, order):
        """
        get a list of all extrema of the array data with order 3
        when the order is greater the number of extrema is less -> not very exact
        :param order: the minimum space between two extremum values
        :param data: 1D array
        :return: list of indexes by which we have a local minimum or maximum
        """
        local_maxima = argrelextrema(data.to_numpy(), np.greater_equal, order=order, mode='clip')[0]
        local_maxima = [local_maxima]
        local_minima = argrelextrema(data.to_numpy(), np.less_equal, order=order, mode='clip')[0]
        local_minima = [local_minima]
        local_maxima.extend(local_minima)
        extrema = np.sort(list(itertools.chain.from_iterable(local_maxima)))
        return extrema
    
    
    def remove_outliers(self,sheet_index,joint_name, number=0.05, order=10):
        """
        

        Parameters
        ----------
        sheet_index : TYPE int
            DESCRIPTION.
        joint_name : TYPE string
            DESCRIPTION.
        number : TYPE float
            DESCRIPTION.0.05to 0.95 or 0.01 to 0.99
        order : TYPE
            DESCRIPTION.

        Returns
        -------
        new_column : TYPE
            DESCRIPTION.

        """
        column = self.df[sheet_index][joint_name]
        indexes = CowsDataset.get_extrema_indexes(column, order)
        sequences = np.split(column.to_numpy(), indexes)
        new_sequences = []
        for i, seq in enumerate(sequences):
            seq = pd.Series(seq)
            seq = seq.mask(~seq.between(seq.quantile(number), seq.quantile(1-number)),np.nan)
            seq = seq.interpolate().to_numpy() 
            new_sequences.append(seq)
        new_column = pd.Series(np.concatenate(new_sequences, axis=0))
        new_column = new_column.interpolate()
        return new_column
            
    def clean_dataset(self,index, number=0.05, order=10):
        columns = CowsDataset.get_joint_names(self.df)
        result = pd.DataFrame()
        for j, joint_name in enumerate(columns):
            column = self.remove_outliers(index,joint_name, number, order)
            result = pd.concat([result, column], axis=1, sort=False)
        result.columns = columns
        return result
    
    
    @staticmethod
    def sliding_window(dataset, side):
        """
        

        Parameters
        ----------
        dataset : TYPE
            DESCRIPTION.

        Returns a part of the dataset with the largest sliding window(between
        1st front-step and 3rd back-step)
                                                                   
        -------
        None.

        """
        periodsf = []
        periodsb = []
        for i, cow in enumerate(dataset):
            df1 = cow['Front_Step'].to_frame().dropna().iloc[:, 0].astype(str).str[0].to_frame().astype(
                int)  # take the first caracter
            test = df1.iloc[0, 0] == 1  # test the begin of the first step
            df1 = df1.diff()
            if test:
                df1.iloc[0] = 1
        
            periodsf.append(np.array(df1.where(df1 == 1).dropna().index + 2))#indexes of the begin of each front step
            #of first step
            
            df2 = cow['Back_Step'].to_frame().dropna().iloc[:, 0].astype(str).str[0].to_frame().astype(
                int)  # take the first caracter
            test = df2.iloc[0, 0] == 1  # test the begin of the first step
            df2 = df2.diff()
            if test:
                df2.iloc[0] = 1
            periodsb.append(np.array(df2.where(df2 == 1).dropna().index + 2))#indexes of the begin of each front step
        periodsf  = np.array(periodsf).T   
        periodsb  = np.array(periodsb).T 
        combined = np.vstack((periodsf, periodsb)).T
        steps = []
        for i in range(9):
            steps.append(np.sort(np.concatenate(combined[i, :])))
        third_bstep = []
        for i, cow_steps in enumerate(steps):
            third_bstep.append(cow_steps[5])#list of the begin of the third back step for each cow
        window_size_side2 = max(third_bstep)
        if (side=='side2'):
            return window_size_side2
        else:
            third_bstep.remove(window_size_side2)
            window_size_side1 =  max(third_bstep)
            return window_size_side1
    
    #define new dataset with window_size rows
    # dataset = pd.DataFrame(dataset.remove_steps_columns())
    # train_dataset = dataset.head(window_size)
        
           
                
        
          
        
        
