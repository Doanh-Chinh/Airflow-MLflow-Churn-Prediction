import os
import pandas as pd
from pandas import DataFrame
import argparse
from tabulate import tabulate
from IPython.display import display
from ydata_profiling import ProfileReport
import sweetviz as sv


class EDA:
    data_path = "../data"
    dataset_name = "BankChurners"
    display_flag = False
    df: DataFrame


    def __init__(self, data_path = "../data", dataset_name = "BankChurners", display_flag = False):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.display_flag = display_flag
        self.df = self.load_data()


    def load_data(self) -> DataFrame:
        data_path = self.data_path
        dataset_name = self.dataset_name
        ext = "csv"
        # Check if the file exists in the data folder
        dataset = f"{dataset_name}.{ext}"
        full_path = os.path.join(data_path, dataset)
        
        if not os.path.exists(full_path):
            print(f"Error: {dataset} not found in {data_path}")
            return

        assert ext == "csv", f"ext = '{ext}' should be csv, otherwise, change other load data method!"
        # Read the CSV file
        df = pd.read_csv(full_path)
        return df

    def generate_profile(self):
        '''
        Generate data report using ProfileReport and sweetviz modules. Return null if report files have already existed.
        '''
        df = self.df
        dataset_name = self.dataset_name

        if os.path.exists(f"./html/{dataset_name}_profiling.html") and os.path.exists(f"./html/{dataset_name}_report.html"):
            print("Report files have already existed.")
            return
        profile = ProfileReport(df, title='Pandas Profiling Report')
        profile.to_file(f"./html/{dataset_name}_profiling.html")
        
        profile.to_notebook_iframe()

        report = sv.analyze(df)
        report.show_html(f"./html/{dataset_name}_report.html", open_browser=False)

    def generate_summary_report(self):
        df = self.df
        dataset_name = self.dataset_name
        display_flag = self.display_flag
        # Initialize a list to collect summary data
        summary_data = []
        # Initialize a list to collect detailed feature summary data
        feature_summary = []
        # Total number of rows and columns
        num_rows, num_cols = df.shape
        
        # Count the total number of null values
        num_null_values = df.isnull().sum().sum()
        
        # Find the number of columns with at least one null value
        null_columns = df.isnull().sum()
        num_columns_with_nulls = (null_columns > 0).sum()
        
        # Append the collected information as a dictionary
        summary_data.append({
            'data': dataset_name,
            'num_cols': num_cols,
            'num_columns_with_nulls': num_columns_with_nulls,
            'num_null_values': num_null_values,
            'num_rows': num_rows
        })

        # Loop through each column (feature) in the dataset
        for column in df.columns:
            num_null_values = df[column].isnull().sum()
            percent_null_values = (num_null_values / num_rows) * 100
            data_type = df[column].dtype
            num_unique_categories = df[column].nunique()

            # Append the feature summary to the list
            feature_summary.append({
                'dataset_name': dataset_name,
                'feature': column,
                'data_type': data_type,
                'num_null_values': num_null_values,
                'percent_null_values': str(round(percent_null_values, 2)) + "%",
                'num_unique_categories': num_unique_categories,
                'total_count': num_rows
            })

        # Convert the summary data into DataFrames
        summary_df = pd.DataFrame(summary_data)
        feature_summary_df = pd.DataFrame(feature_summary)
        def printS():
            '''
            Print the summary headline
            '''
            print("-"*20 + "Summary".upper() + "-"*20)
        def printD():
            '''
            Print the detail headline
            '''
            print("-"*20 + "Detail".upper() + "-"*20)

        if display_flag:
            # Print the summary and feature overview using display 
            printS()       
            display(summary_df)
            printD()
            display(feature_summary_df)
        else:
            # Print the summary and feature overview using tabulate
            printS()
            print(tabulate(summary_df, headers='keys', tablefmt='fancy_grid', showindex=False))
            printD()
            print(tabulate(feature_summary_df, headers='keys', tablefmt='fancy_grid', showindex=False))
    def eda_pipeline(self):
        self.generate_profile()
        self.generate_summary_report()
        print("Complete EDA pipeline.")
    
    def drop_columns(self, *cols):
        """
        Remove unnecessary columns
        """
        df = self.df
        ls_cols = [col for col in cols]
        print(ls_cols)
        df.drop(labels=ls_cols, axis=1, inplace=True, errors='ignore')
        self.df = df  


if __name__ == "__main__":

    # Parse the terminal argument for dataset names
    parser = argparse.ArgumentParser(description="Generate a data summary for given datasets.")
    parser.add_argument("--data_path", type=str, default='../data', help="Folder path directs to data file")
    parser.add_argument("--dataset_name", type=str, help="Dataset name (without .csv extension)")
    
    # Get the dataset names from the command-line arguments
    args = parser.parse_args()
    data_path = args.data_path
    dataset_name = args.dataset_name
    eda = EDA()
    # eda.eda_pipeline()
    data = eda.df
    eda.drop_columns('CLIENTNUM', data.columns[-1], data.columns[-2])
    print(eda.df.head(1).T)
    # python .\eda_pipeline.py --data_path ../data --dataset_name BankChurners
    
