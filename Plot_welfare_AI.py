"""
@author: Yasmine
"""
import pandas as pd
import matplotlib.pyplot as plt


def plot_step_start(df, step_type, color):
    """
    Plot vertical lines at the begin of every step
    :param df: Data frame
    :param step_type: Front_Step or Back_Step
    :param color: Color of the vertical line

    """
    # Determine the indexes where the value of Front_Step changes

    df1 = df[step_type].to_frame().dropna().iloc[:, 0].astype(str).str[0].to_frame().astype(int)# take the first caracter
    test = df1.iloc[0, 0] == 1 # test the begin of the first step
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
                 s=step_type + ' ' + str(i + 1), color=color, fontsize=15, rotation='vertical')
    return periods


def plot_all_markers_subplots(df, sheet):
    """
    plot all markers of the cow's body as subplots
    and mark the begin of every step
    :param df: Data frame
    :param sheet: Excel sheet

    """
    columns = df.iloc[0, 2:34].index
    fig, axes = plt.subplots(9, 4, figsize=(50, 35))
    fig.suptitle(sheet, fontsize=60)
    line = plt.Line2D((0, 1), (0.455, 0.455), color="k", linewidth=7)
    fig.add_artist(line)
    plt_cols = 4
    plt_rows = 8

    coordinates = ['X', 'Y', 'Z', 'R']

    for i, coordinate in enumerate(coordinates):
        plt.subplot(plt_rows + 1, plt_cols, i+1)
        plt.text(0.5, 0.2, horizontalalignment='center',verticalalignment='center', s=coordinate, fontsize= 50)
        plt.axis('off')
    for i, column in enumerate(columns):
        # plt.subplot(plt_rows + 1, plt_cols, i + 7 + int(i/4))
        plt.subplot(plt_rows + 1, plt_cols, i + 5)
        plt.plot(df[column])
        plot_step_start(df, 'Front_Step', 'red')
        plot_step_start(df, 'Back_Step', 'green')
        plt.xlabel(' ', fontsize=30)
        plt.ylabel('\n\n'+column, fontsize=30)
    fig.text(0, 0.65, s='Front leg', rotation='vertical', fontsize=50)
    fig.text(0, 0.15, s='Back leg', rotation='vertical', fontsize=50)
    plt.show()


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
        plot_step_start(df, 'Front_Step', 'red')
        plot_step_start(df, 'Back_Step', 'green')
        plt.ylabel(column)
        plt.title(column + '  ' + sheet)
        plt.show()


def main():
    """
    Plots of the Markers for all the sheets of the Excel file 'ScaledCoordinates_Post-Trial.xlsx'
    """
    sheets = ['Scaled-Coord_2063_Side1', 'Scaled-Coord_2063_Side2',
              'Scaled-Coord_5870(2)_Side2', 'Scaled-Coord_5870(2)_Side1',
              'Scaled-Coord_2078(2)_Side1', 'Scaled-Coord_2078(2)_Side2',
              'Scaled-Coord_5327(2)_Side1', 'Scaled-Coord_5327(2)_Side2',
              'Scaled-Coord_8527_Side1', 'Scaled-Coord_8527_Side2',
              'Scaled-Coord_8531_Side1', 'Scaled-Coord_8531_Side2',
              'Scaled-Coord_2066_Side1', 'Scaled-Coord_2066_Side2',
              'Scaled-Coord_5871_Side1', 'Scaled-Coord_5871_Side2',
              'Scaled-Coord_5865_Side1', 'Scaled-Coord_5865_Side2', ]
    for sheet in sheets:
        df = pd.read_excel(r"D:\BA_Yasmine_UQAM\Plot\ScaledCoordinates_Post-Trial.xlsx", sheet)
        plot_all_markers_subplots(df, sheet)



if __name__ == '__main__':
    main()