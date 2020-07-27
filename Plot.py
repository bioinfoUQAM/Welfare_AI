import matplotlib.pyplot as plt
from Welfare_AI_dataset import CowsDataset
import os


class Plot_dataset:
    def __init__(self):
        pass

    @staticmethod
    def plot_step_start(df, step_type, color, fontsize):
        """
        Plot vertical lines at the begin of every step
        :param df: Data frame
        :param step_type: Front_Step or Back_Step
        :param color: Color of the vertical line

        """
        # Determine the indexes where the value of Front_Step changes

        df1 = df[step_type].to_frame().dropna().iloc[:, 0].astype(str).str[0].to_frame().astype(int)  # take the first caracter
        test = df1.iloc[0, 0] == 1  # test the begin of the first step
        df1 = df1.diff()
        if test:
            df1.iloc[0] = 1

        periods = df1.where(df1 == 1).dropna().index + 2

        # Plot the red vertical lines
        for index in periods:
            plt.axvline(index, ymin=0, ymax=1, color=color)

        # Plot the Record Text
        y_lim = plt.gca().get_ylim()
        for i, period in enumerate(periods):
            plt.text(y=(y_lim[0] + y_lim[1]) / 2, x=period + 5,
                     s=step_type + ' ' + str(i + 1), color=color, fontsize=fontsize, rotation='vertical')
        return periods

    @staticmethod
    def plot_all_markers_subplots(df, sheet):
        """
        plot all markers of the cow's body as subplots
        and mark the begin of every step
        :param df: Data frame
        :param sheet: Excel sheet

        """
        columns = df.iloc[0, 2:34].index
        fig, axes = plt.subplots(9, 4, figsize=(50, 35))
        cow_name = CowsDataset.get_cow_name(sheet)
        side = sheet[-5:]
        title = cow_name + '  ' + side
        fig.suptitle(title, fontsize=60)
        line = plt.Line2D((0, 1), (0.45, 0.45), color="k", linewidth=7)
        fig.add_artist(line)
        plt_cols = 4
        plt_rows = 8

        coordinates = ['X', 'Y', 'Z', 'R']

        for i, coordinate in enumerate(coordinates):
            plt.subplot(plt_rows + 1, plt_cols, i + 1)
            plt.text(0.5, 0.2, horizontalalignment='center', verticalalignment='center', s=coordinate, fontsize=50)
            plt.axis('off')
        for i, column in enumerate(columns):
            # plt.subplot(plt_rows + 1, plt_cols, i + 7 + int(i/4))
            plt.subplot(plt_rows + 1, plt_cols, i + 5)
            plt.plot(df[column])
            Plot_dataset.plot_step_start(df, 'Front_Step', 'red', 15)
            Plot_dataset.plot_step_start(df, 'Back_Step', 'green', 15)
            plt.xlabel(' ', fontsize=30)
            plt.ylabel('\n\n' + column, fontsize=30)
        fig.text(0, 0.65, s='Front leg', rotation='vertical', fontsize=50)
        fig.text(0, 0.15, s='Back leg', rotation='vertical', fontsize=50)
        image_name = cow_name + '_motionvis_' + side +'.png'
        plt.gcf().savefig(os.path.join(os.path.dirname(__file__), image_name))
        plt.show()



    @staticmethod
    def plot_all_markers(df, sheet):
        """
        plot all markers of the cow's body separately
        and mark the begin of every step
        :param df: Data frame
        :param sheet: Sheet of the Excel file

        """
        columns = df.iloc[0, 2:34].index
        for column in columns:
            plt.plot(df[column])
            Plot_dataset.plot_step_start(df, 'Front_Step', 'red', 15)
            Plot_dataset.plot_step_start(df, 'Back_Step', 'green', 15)
            plt.ylabel(column)
            plt.title(column + '  ' + sheet)
            plt.show()
    
    
    @staticmethod
    def get_max_scale_value(dataset, column):
        """
        

        Parameters
        ----------
        dataset : TYPE CowsDataset
            DESCRIPTION.
        column : TYPE string
            DESCRIPTION. column_name

        Returns the maximum value of a given column for the different cleaned 
        cows data
        -------
        TYPE
            DESCRIPTION.

        """
        n_cows = 9
        max_list = [max(dataset.remove_outliers(i,column, number=0.05, order=10).dropna()) for i in range(n_cows)]#list with the maxima of each column for each cow
        return max(max_list)
    
     
    @staticmethod
    def get_min_scale_value(dataset, column):
        """
        

        Parameters
        ----------
        dataset : TYPE CowsDataset
            DESCRIPTION.
        column : TYPE string
            DESCRIPTION.column_name

        Returns the minimum value of a given column for the different cleaned 
        cows data
        -------
        TYPE
            DESCRIPTION.

        """
        n_cows = 9
        min_list = [min(dataset.remove_outliers(i,column, number=0.05, order=10).dropna()) for i in range(n_cows)]#list with the minima of each column for each cow
        return min(min_list)
        
    @staticmethod
    def plot_subplots_joint(dataset):
        """
        

        Parameters
        ----------
        dataset : TYPE
            DESCRIPTION.

        Returns change side1 and side2
        -------
        None.

        """
        joint_names = CowsDataset.get_joint_names(dataset)
        list_of_joints = dataset.get_list_of_joints()
        for i, column in enumerate(joint_names):
            fig, axes = plt.subplots(3, 3, figsize=(15,8),sharey='col')
            side = dataset.sheet_names[0][-5:]
            title = column + '  ' + side
            fig.suptitle(title, fontsize=20)
            min = Plot_dataset.get_min_scale_value(dataset, column)
            max = Plot_dataset.get_max_scale_value(dataset, column)
            for j, ax in enumerate(axes.flat):
                cow_name = CowsDataset.get_cow_name(dataset.sheet_names[j])
                plt.subplot(3, 3, j+1)
                ax = plt.plot(dataset.remove_outliers(j,column, number=0.05, order=10))
                plt.ylim(min, max)
                plt.title("\n\n" + cow_name)
                plt.subplots_adjust(wspace=3)
                Plot_dataset.plot_step_start(dataset[j], 'Front_Step', 'red', 7)
                Plot_dataset.plot_step_start(dataset[j], 'Back_Step', 'green', 7)
            fig.tight_layout()
            image_name = column + '_motionvis_' + '.png'
            plt.gcf().savefig(os.path.join(os.path.dirname(__file__),'Plot', 'joints','side2', image_name))
            plt.show()
