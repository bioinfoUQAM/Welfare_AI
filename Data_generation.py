import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from numba.experimental import jitclass
from scipy.signal import medfilt
import random
from matplotlib.legend_handler import HandlerLine2D
from scipy.ndimage.interpolation import shift
import itertools
from scipy.signal import argrelextrema
from Welfare_AI_dataset import CowsDataset
from scipy.ndimage import gaussian_filter
from numba import jit, cuda, float32, int32
from tqdm import tqdm
import xlsxwriter


# spec = [
#     ('value', int32),               # a simple scalar field
#     ('array', float32[:]),          # an array field
# ]


#@jitclass(spec)
class GeneratedData:
    def __init__(self, value):
        # self.value = value
        # self.array = np.zeros(value, dtype=np.float32)
        pass

    @staticmethod
    def generate_samples(dataset, sheet_index, column_name):
        """
        generate new sample of column(joint) in the data
        by adding a random value between [-2%,+2%] to the original values
        :param dataset: Object of class CowsDataset
        :param sheet_index: the index (beginning from 0) of the sheet of the excel file
        :param column_name: String The name of the joint
        :return: new synthetic data based on the real data
        """
        generated_data = []
        movement_begin = np.where(~(np.isnan(dataset[sheet_index][column_name])))# to see where the movement begins
        nan_indexes = np.where((np.isnan(dataset[sheet_index][column_name])))
        #print(movement_begin)
        joint_column = dataset[sheet_index][column_name]
        joint_column = CowsDataset.remove_outliers(dataset, sheet_index,column_name, number=0.05, order=10)#remove outliers
        joint_column = joint_column.round(3)# round with 3 decimal numbers
        # std_deviation = np.std(joint_column)
        # sigma = 3*std_deviation
        kernel_size = 9
        for i, row in enumerate(joint_column):
            if np.isnan(row):
                generated_data.append(row)
            else:
                b_min = int(min(-0.02*row, 0.02*row))
                b_max = -b_min
                #random.seed(i)
                generated_data.append(row + random.randint(b_min, b_max))
                #generated_data = pd.DataFrame(generated_data)
        #generated_data = shift(medfilt(generated_data), shift=movement_begin, cval=np.NaN)# I don't know why il doesn't work with movement_begin[0]
        #for the first value( je crois le problème est là!!)
        generated_data = medfilt(generated_data, kernel_size = kernel_size)
        #generated_data = gaussian_filter(generated_data, sigma)
        # line1, = plt.plot(generated_data, label='generated_data' + column_name)
        # line2, = plt.plot(joint_column, label='real_data' + column_name)
        # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
        # plt.show()
        # ax = joint_column.plot.hist(bins=15)
        # plt.show()
        return generated_data

    # @staticmethod
    # def generated_data_to_excel(df, file_name, sheet_name):
    #      writer = pd.ExcelWriter(file_name +'.xlsx', engine='xlsxwriter')
    #     # Convert the dataframe to an XlsxWriter Excel object.
    #      df.to_excel(writer, sheet_name=sheet_name, index=False)
    #     # Close the Pandas Excel writer and output the Excel file.
    #      writer.save()

    @staticmethod
    def create_generated_cow_data(dataset, sheet_index, n_samples):
        """
        generate new samples of columns(joints) of one sheet(cow)
        :param n_samples: The number of synthetic samples to generate
        :param dataset: object from class Cowsdataset
        :param sheet_index: the index (beginning from 0) of the sheet of the excel file
        :return: all the new samples of the different columns/joints of the cow
        """
        file_name = CowsDataset.get_cow_names()[sheet_index]
        writer = pd.ExcelWriter(file_name+'_Medfilt7%.xlsx', engine='xlsxwriter')
        sheets = []
        new_dataset = []
        for i in range(n_samples):
            sheet_name = file_name #+ str(i)
            sheets.append(sheet_name)
            generated_data = []
            for j, column in enumerate(dataset[sheet_index]):
                generated_data.append(GeneratedData.generate_samples(dataset, sheet_index, column))
            df = pd.DataFrame(generated_data).T
            df.columns = CowsDataset.get_joint_names(dataset)# Convert the dataframe to an XlsxWriter Excel object.
            new_dataset.append(df)
            #df.to_excel(writer, sheet_name=sheet_name, index=False)
                #GeneratedData.generated_data_to_excel(df, file_name, sheet_name)
        return new_dataset, sheets
        #writer.save()


    @staticmethod
    def find_monotone_sequences(x):
        """
        find all the monotone sequences of an array( with every little vibration)-> very exact
        :param x: 1D array
        :return: dictionary with lists of monotone sequences
        """
        result = {"increasing": [],
                  "equal": [],
                  "decreasing": [],
                  }

        def two_item_relation(prev, curr):  # compare two items in list, results in what is effectively a 3 way flag
            if prev <= curr:
                return "increasing"
            elif prev == curr:
                return "equal"
            else:
                return "decreasing"

        prev_state = two_item_relation(prev, curr)  # keep track of previous state
        result[prev_state].append([prev])  # handle first item of list

        x_shifted = iter(x)
        next(x_shifted)  # x_shifted is now similar to x[1:]

        for curr in x_shifted:
            curr_state = two_item_relation(prev, curr)
            if prev_state == curr_state:  # compare if current and previous states were same.
                result[curr_state][-1].append(curr)
            else:  # states were different. aka a change in trend
                result[curr_state].append([])
                result[curr_state][-1].extend([prev, curr])
            prev = curr
            prev_state = curr_state
        print(result)
        return result

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

    @staticmethod
    def generate_monotone_sequences(data, order):
        """
        generate randomly new samples of a column(joints) according to the mean value an the variance
        of every monotone sequence of the data
        :param order: the minimum space between two extremum values
        :param data: 1D array
        :return:new generated 1D array
        """
        random_samples = []
        sequences = GeneratedData.get_extrema_indexes(data, order)
        for i, sequence in enumerate(sequences):
            if (i+1) < len(sequences):
                param = scipy.stats.norm.fit(data[sequences[i]+1: sequences[i+1]+1])# get the parameters
                # of the sequence of data
                random_samples.extend(np.sort(scipy.stats.norm.rvs(param[0], param[1], size=data.dropna().shape[0])))#generate
                # new sequence based on the parameters of the original sequence and put it in a list

            else:
                pass

        plt.plot(random_samples)
        plt.plot(data)
        plt.show()
        ax = data.plot.hist(bins=10)



