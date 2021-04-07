import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
# from pprint import pprint


def get_api():

    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)

    client = gspread.authorize(creds)

    sheet = client.open("test").sheet1  # Open the spreadhseet

    data = sheet.get_all_records()  # Get a list of all records
    # pprint(data)

    row = sheet.row_values(3)  # Get a specific row
    # print(row)

    col = sheet.col_values(3)  # Get a specific column
    # print(col)
    cell = sheet.cell(2,3).value  # Get the value o a specific cell
    combos_new_sheet = client.open("test").worksheet('combos_new_test')
    combos_new = combos_new_sheet.get_all_records()
    df = pd.DataFrame(combos_new)
    combos_matrix1 = df.replace(r'^\s*$', np.nan, regex=True)
    combos_desc_sheet = client.open("test").worksheet('combos_desc')
    combos_desc = combos_desc_sheet.get_all_records()
    df = pd.DataFrame(combos_desc)
    combos_desc1 = df.replace(r'^\s*$', np.nan, regex=True)

    matrix_sheet = client.open("test").worksheet('matrix')
    matrix = matrix_sheet.get_all_records()
    matrix1 = pd.DataFrame(matrix)


    # insertRow = ["hello", 5, "red", "blue"]
    # sheet.add_rows(row, 4)  # Insert the list as a row at index 4

    # sheet.update_cell(2,2, "CHANGED")  # Update one cell

    # numRows = sheet.row_count  # Get the number of rows in the sheet
    return(cell, combos_matrix1, combos_desc1)






# print(get_api())