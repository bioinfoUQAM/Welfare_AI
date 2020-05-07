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
    df1 = df[step_type].to_frame().dropna()
    periods = df1.iloc[:, 0].str.contains("Start").dropna().to_frame().index

    # Plot the red vertical lines
    for index in periods:
        plt.axvline(index, ymin=0, ymax=1, color=color)

    # Plot the Record Text
    y_lim = plt.gca().get_ylim()
    for i, period in enumerate(periods):
        plt.text(y=(y_lim[0] + y_lim[1]) / 2, x=period + 5,
                 s=step_type + ' ' + str(i + 1), color=color, fontsize=15, rotation='vertical')


def plot_all_markers_subplots(df, sheet):
    """
    plot all markers of the cow's body as subplots
    and mark the begin of every step
    :param df: Data frame
    :param sheet: Excel sheet

    """
    columns = df.iloc[0, 2:34].index
    fig = plt.gcf()
    fig.set_size_inches(43.2, 28.8)
    size = fig.get_size_inches() * fig.dpi
    print(size)
    fig.suptitle(sheet, fontsize=60)
    for i, column in enumerate(columns):
        plt.subplot(9, 4, i + 5)
        plt.plot(df[column])
        plot_step_start(df, 'Front_Step', 'red')
        plot_step_start(df, 'Back_Step', 'green')
        plt.ylabel(column, fontsize=30)
    fig.tight_layout()
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
    sheets = ['Scaled-Coord_2063_Side2', 'Scaled-Coord_5870(2)_Side2',
              'Scaled-Coord_2078(2)_Side1', 'Scaled-Coord_2078(2)_Side2',
              'Scaled-Coord_5327(2)_Side1', 'Scaled-Coord_5327(2)_Side2']
    for sheet in sheets:
        df = pd.read_excel(r"D:\BA_Yasmine_UQAM\Plot\ScaledCoordinates_Post-Trial.xlsx", sheet)
        plot_all_markers(df, sheet)


if __name__ == '__main__':
    main()
