from string import ascii_uppercase

import pandas as pd


class DictMapper:
    """
    A class to create mapping dictionaries from Excel data.

    Attributes:
        file_path (str): The path to the Excel file.
        sheet_name (str): The name of the sheet to load.
        df (pd.DataFrame): The loaded DataFrame.
        df_data (pd.DataFrame): The cleaned DataFrame without unnecessary rows.
    """
    def __init__(self, file_path: str, sheet_name: str):
        """
        Initializes the DictMapper with the file path and sheet name.

        Args:
            file_path (str): The path to the Excel file.
            sheet_name (str): The name of the sheet to load.
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.df = None
        self.df_data = None
        self.load_data()

    def load_data(self):
        """
        Loads data from the Excel file and renames and cleans columns.
        """
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        self._rename_columns()
        self._clean_data()

    def _rename_columns(self):
        """
        Renames columns in the DataFrame to single letters (A, B, C, ...).
        """
        list_new_name = [letter for letter in ascii_uppercase[: len(self.df.columns)]]
        rename_dict = dict(zip(self.df.columns, list_new_name))
        self.df.rename(columns=rename_dict, inplace=True)

    def _clean_data(self):
        """
        Cleans the DataFrame by dropping the first row and rows with NaN in column 'C'.
        """
        self.df_data = self.df.iloc[1:].copy()  # Drop the first row if unnecessary
        self.df_data.dropna(
            subset=["C"], inplace=True
        )  # Drop rows with NaN in column 'C'

    def create_mapping_dict(self, source_columns: list, target_column: str):
        """
        Creates a mapping dictionary from specified source columns to a target column.

        Args:
            source_columns (list): List of columns to use as keys.
            target_column (str): Column to use as values.

        Returns:
            dict: Mapping dictionary.
        """
        mapping_dict = {}
        for col in source_columns:
            col_dict = pd.Series(
                self.df_data[target_column].values, index=self.df_data[col]
            ).to_dict()
            mapping_dict.update(col_dict)
        # Remove NaN keys if any
        mapping_dict = {k: v for k, v in mapping_dict.items() if pd.notna(k)}
        return mapping_dict

    def get_data(self):
        """
        Returns the cleaned DataFrame.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        return self.df_data
