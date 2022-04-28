#!/usr/bin/env python
# coding: utf-8

# ### Scripts Combined
# <body>
#     <ul>
#         <li>cleaner_data_formatting_multiple_channels_Hora</li>
#         <li>data_formatting_Hora</li>
#     </ul>
# </body>
# 
# ### Characteristics of the Script
# <body>
#     <ul>
#         <li>Reads data from given filename.</li>
#         <li>Figures out the location of the data on the spreadsheet.</li>
#         <li>Documents if the data are transposed- ignoring everything inside of the soon-to-be dataframes.</li>
#         <li>Extracts channel names.</li>
#         <li>Adds dataframes of each channel iteratively to each other using indexing from the second point.</li>
#         <li>Takes into account the channel column for transposing the data- working with everything inside the dataframes.</li>
#         <li>Makes one large dataframe with all channels.</li>
#         <li>Filters out wells that are not used</li>
#         <li>Replaces "OVER" saturated values with Nan.</li>
#         <li>Writes an Excel spreadsheet in the format read by the plotting script- Data sheet with data, Wells sheet requiring user input.</li>
#     </ul>
# </body>
# 
# ### Improvement to combining_notebooks.ipynb
# This script allows for user inputs. If the user provides only an input file name, the program will implicitly save the processed data in the same directory as the input file. An output directory can be specified where the processed file will be stored. In both cases, the name of the processed data starts with processed_ + [name of input file]

# In[2]:


## Goal of the Chunk: Load packages and other necessary functions

# I will import the same packages from the data processing file to ensure I have everything that I need
import sys
import os
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
import openpyxl
import pandas as pd
import plotnine
from plotnine import *
# %matplotlib inline 
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = "{:,.4f}".format # display float values to four decimal places
#modify some matplotlib parameters to manage the images for illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# In[3]:


## Goal of the Chunk: Specify file to be opened and load its information

# Test inputs for now
# The user has two options:
# 1. Input the file path for the data to be transformed and supply a directory for the output file to be saved in.
# 2. Input the file path for the data to be transformed and supply an empty string.
#    The empty string means that it is assumed that the user wants to store their output data in the same input directory.
# In the script format, the inputs will be in the form of system arguments in the command line.
# The sample lines will take the following format:
# $ python temp.py /myfileishere/file.xlsx /outputfolder/
# sys.argv[1]


## This concept was borrowed from the Jupyter Notebook script for finding genes of interest that July and I wrote.
# If no "True" is found, this part stops the rest of the code from running.
# It then informs the user if the gene symbol was entered incorrectly and stops running the subsequent code.
class StopExecution(Exception):
    def _render_traceback_(self):
        pass

    
Found = False

# Option 1
if len(sys.argv) == 3:
    # Both the input file name is valid and the output directory exists
    if os.path.isfile(sys.argv[1]) == True and os.path.isdir(sys.argv[2]) == True:
        # Splits the file name, called input_file, from the rest of the path
        input_dir, input_file = os.path.split(sys.argv[1])
        # places the word "processed_" before the original file name to indicate the data have been modified.
        output_path = os.path.join(sys.argv[2], "processed_" + input_file)
        Found = True
    # The input file name is valid but the output directory is yet to exist
    elif os.path.isfile(sys.argv[1]) == True and os.path.isdir(sys.argv[2]) == False:
        output_dir_check_above, output_dir_check = os.path.split(sys.argv[2])
        # Taking everything above where the new directory is created and separating from the folder that will be created
        if os.path.isdir(output_dir_check_above) == True:
            input_dir, input_file = os.path.split(sys.argv[1])
            # Make a new directory and storing the file there with previously spliced information
            new_dir = os.path.join(output_dir_check_above, output_dir_check)
            os.mkdir(new_dir)
            # Saving the file in the new directory
            output_path = os.path.join(new_dir, ("processed_" + input_file))
            Found = True
        # Handles an incorrect path above where the new directory wants to be made
        else:
            print("The directory supplied does not have a valid path")
            pass
    # Handles incorrect input file- completely incorrect path supplied by at least the input, possibly the output too
    else:
        print("Either the input file name or the output directory name was entered incorrectly.")
# Option 2
elif len(sys.argv) == 2:
    # Saving the output file in the same directory as the valid input file provided
    if os.path.isfile(sys.argv[1]) == True:
        input_dir, input_file = os.path.split(sys.argv[1])
        output_path = os.path.join(input_dir, "processed_" + input_file)
        Found = True
    # Handles an invalid input
    else:
        print("Error: The input file name was entered incorrectly.")
### These parts stay the same as the interactive script
# Too many arguments
elif len(sys.argv) > 3:
    print("Error: Too many arguments given. There should be a maximum of two.")
# Not enough arguments given
else:
    print("Error: Not enough argument(s) given.")
    print("There should be at least one correct file path with the option of a new directory given.") 
    
    
# Stopping the rest of the script from running if there are invalid inputs
if Found == False:
    print("The script was stopped.")
    raise StopExecution

# For the sake of clarity and not using "sys.argv[1]" for the rest of the document, I will call it input_file_path
input_file_path = sys.argv[1]
# Reading in the data from the required input   
data_imported = pd.read_excel(input_file_path)


# In[6]:


## Goal of the Chunk: Extract the location of where the data first appear- using "Start Time" as a landmark

# Adapted from: https://thispointer.com/python-find-indexes-of-an-element-in-pandas-dataframe/
# Creating an empty list to store the results
position_list = list()
# Get bool dataframe with True at positions where the given value exists
# Only two cells will have True where "Start Time" is, the rest will be false
result = data_imported.isin(["Start Time"])
# Get list of columns that contains the value
# Only col_1 has 2 "Start Time" values, so it will be the only one with a True
series_obj = result.any()
# Returns the only column with a True containing "Start Time", col_1
column_names = list(series_obj[series_obj == True].index)
# Iterate over list of columns and fetch the rows indexes where value exists
# aka this will go row by row in col_1 looking for "Start Time"
for col in column_names:
    # This isolates the rows where "Start Time is"
    rows = list(result[col][result[col] == True].index)
    # Rows is a list of the rows with "Start Time"
    # The next loop appends their position to the original position_list
    for row in rows:
        position_list.append((row, col))


# In[7]:


## Goal of the Chunk: Extract the location of where the data stop- using "End Time" as a landmark.

# Pretty much the same thing as "Start Time"
# Adapted from: https://thispointer.com/python-find-indexes-of-an-element-in-pandas-dataframe/
# Creating an empty list to store the results
position_list_end = list()
# Get bool dataframe with True at positions where the given value exists
# Only two cells will have True where "Start Time" is, the rest will be false
result_end = data_imported.isin(["End Time"])
result_end
# Get list of columns that contains the value
# Only col_1 has 2 "Start Time" values, so it will be the only one with a True
series_obj_end = result_end.any()
# Returns the only column with a True containing "Start Time", col_1
column_names_end = list(series_obj_end[series_obj_end == True].index)
# Iterate over list of columns and fetch the rows indexes where value exists
# aka this will go row by row in col_1 looking for "Start Time"
for col_end in column_names_end:
    # This isolates the rows where "End Time is"
    rows_end = list(result_end[col_end][result_end[col_end] == True].index)
    # Rows is a list of the rows with "Start Time"
    # There is a difference between the location here and the location in Excel.
    # For example, position 34 is this script is equivalent to position 36 in Excel.
    # When the Excel file is read in, the first row becomes the header name (accounting for 1 location difference).
    # Python starts its index at 0 while Excel starts at 1 (accounting for the other 1 location difference). 
    # As long as the position in Excel is +2 compared to the position in here, everything is working correctly.
    # A difference other than +2 means something is not working correctly.
    # The next loop appends their position to the original position_list
    for row_end in rows_end:
        position_list_end.append((row_end, col_end))


# In[12]:


## Goal of the Chunk: Remove everything around where the data start (including two rows above it). 
## Document where the data end exactly.

# There are three rows in between the second "Start Time"
skip_start = position_list[1][0] + 3
# There is one row in between the last of the data and "End Time"
# Skipping the number of rows in skip_start affects the indexing for skip_end
skip_end = position_list_end[1][0] - skip_start - 1

# # Extracting the channel name
data_imported_first_channel = data_imported.rename(columns={data_imported.columns[0]: 'col_1'})
skip_first_channel = skip_start - 1
data_imported_first_channel["col_1"][skip_first_channel]

data_imported = pd.read_excel(input_file_path)

# Extracting the first channel name
data_imported_first_channel_name = data_imported.rename(columns={data_imported.columns[0]: 'col_1'})
skip_first_channel_name = skip_start - 1
first_channel = data_imported_first_channel_name["col_1"][skip_first_channel_name]
# first_channel

# Reading in Excel file skipping all the header information
data_imported_with_skips = pd.read_excel(input_file_path, header = skip_start - 2)

# Unselecting rows that aren't wells
data_imported_with_skips = data_imported_with_skips[:skip_end]


# In[13]:


## Goal of the Chunk: Determine if wells are columns (right format) or if wells are rows (wrong format)
true_counter = 0

data_imported_with_skips_test = data_imported_with_skips

data_imported_with_skips_test.columns = data_imported_with_skips_test.iloc[2]

# counting the number of times "Time [s]" appears as a column (correct format)
for col in data_imported_with_skips_test:
    if col == "Time [s]":
        # There should only be one "Time [s]"
        true_counter += 1
    else:
        pass


# In[14]:


## Goal of the Chunk: Either allow the skip chunk to pass as is (correct format) 
## or force a transformation over the entire chunk (incorrect format).
## Then, there is a name-change that calls the first column "Potential_Channel_Names"

# When Time [s] appears as a row (wrong format), the true_counter == 0
if true_counter == 0:
    # Transposes the data
    data_imported_t = data_imported_with_skips[2:]
    data_imported_t = data_imported_t.T
    data_imported_t = data_imported_t.reset_index(drop = True)
    data_imported_t.columns = data_imported_t.iloc[0]
    data_imported_t = data_imported_t[1:]
#     data_imported_t["Cycle Nr."] = data_imported_t["Cycle Nr."].astype(int)
    data_imported_with_skips_tp = data_imported_t
else:
    # Leaves the data as are, renaming the dataframe in the process
    data_imported_with_skips_tp = data_imported_with_skips


# Deselects the first row that was used to make the column names
# data_imported_with_skips_channel = data_imported_with_skips.rename(columns={data_imported_with_skips_tp.columns[0]: 'Channel'})
data_imported_with_skips_channel_renamed = data_imported_with_skips.rename(columns={data_imported_with_skips.columns[0]: 'Potential_Channel_Names'})


# In[15]:


## Goal of the Chunk: Extract the channel name(s) from the previous chunk using the "Potential_Channel_Name" column

# Creating a list to store values in- already including the first channel
channels_available = []


# The counter prevents the while loop from running forever- because it will run as long as the statement is true
# Establishing a counter variable
counter = 0
# Will run through these steps as long as the rest of the file is
while counter < (len(data_imported_with_skips_channel_renamed["Potential_Channel_Names"])):
    # Looking at each row in the "Cycle Nr." column
    for row in data_imported_with_skips_channel_renamed["Potential_Channel_Names"]:
        # A try except loop is super useful
        # if you try to convert a string to an integer (int(string)), then the program will throw an error
        # Instead of the error breaking the code, this loop continues and filters out non error (try) vs error (except)
        # I don't care about the cycle numbers, so those are ignored if the row is an integer
        try:
            row_no_letter = row[1:]
            row_no_letter = int(row_no_letter)
            pass
        except:
            channels_available.append(row)
        counter += 1
    
# Filtering out spaces as a possibility    
channels_available_non_nan = []
    
# Resetting the counter from the previous loop to work for the following loop    
counter = 0
while counter < (len(channels_available)):
    # Looking at each row in the "Cycle Nr." column
    for channel in channels_available:
        # Similar process as before, now removing the spaces that are stored as na values 
        try:
            nan_removal =  pd.isna(float(channel))
        except:
            # appending all non na values
            channels_available_non_nan.append(channel)
        counter += 1
    

# Resetting the list to be blank and include only the finalized values
channels_available = []

for channel in channels_available_non_nan:
    if channel == "Cycle Nr.":
        # Ignoring the column "Cycle Nr."
        pass
    elif channel == "Time [s]":
        # Ignoring the column Time [s] for now
        pass
    elif channel == "Temp. [°C]":
        # Ignoring the column Temp. [°C]
        pass
    elif type(channel) == float:
        # Ignoring any other types of values that are potentially numbers
        pass
    else:
        # For everything that is left- e.g. OD600, sfGFP_Gain60, Ruby, it is appended to the final list
        channels_available.append(channel)


# In[16]:


## Goal of the Chunk: Determine the indices of each channel (start only)
# Adapted from: https://thispointer.com/python-find-indexes-of-an-element-in-pandas-dataframe/
# Creating an empty list to store the results
position_list_channels = list()
# Get bool dataframe with True at positions where the given value exists
# Only two cells will have True where "Start Time" is, the rest will be false
result_channels = data_imported_with_skips_channel_renamed.isin(channels_available)
# Get list of columns that contains the value
# Only "Cycle Nr." and "Channel" has the channel names, so it will be the only one with a True
series_obj_channels = result_channels.any()
# Returns the only columns with a True containing channel names, "Cycle Nr." and "Channel"
column_names_channels = list(series_obj_channels[series_obj_channels == True].index)
# Iterate over list of columns and fetch the rows indexes where value exists
# aka this will go row by row in col_1 looking for "Start Time"
for col in column_names_channels:
    # This isolates the rows where channel names are
    rows_channels = list(result_channels[col][result_channels[col] == True].index)
    # Rows is a list of the rows with "Start Time"
    # The next loop appends their position to the original position_list
    for row_channels in rows_channels:
        position_list_channels.append((row_channels, col))


# In[18]:


## Goal of the Chunk: Determine the indices of each channel (end only)

data_imported_blank_space = pd.read_excel(input_file_path, header = skip_start + 1)

last_end_time = []
# I should add in where it stops- at the last blank space
# The counter prevents the while loop from running forever- because it will run as long as the statement is true
# Establishing a counter variable
counter = 0
# Will run through these steps as long as the rest of the file is
while counter < (len(data_imported_blank_space["Cycle Nr."])):
    # Looking at each row in the "Cycle Nr." column
    for row in data_imported_blank_space["Cycle Nr."]:
        # A try except loop is super useful
        # if you try to convert a string to an integer (int(string)), then the program will throw an error
        # Instead of the error breaking the code, this loop continues and filters out non error (try) vs error (except)
        # I don't care about the cycle numbers, so those are ignored if the row is an integer
        try:
            row = int(row)
            pass
        except:
            # Ignoring the "Cycle Nr." part
            if row == "Cycle Nr.":
                pass
            # Ignoring the blank spaces, which are coded as floats
            elif type(row) == float:
                pass
            # Appending each channel to the list
            elif row == "End Time":
                last_end_time.append(counter)
        # No matter if the row goes into the try or except, it will increase the counter value by each iteration
        counter += 1


# In[19]:


## Goal of the Chunk: Storing the indices with each channel name, adjusting for the header at the beginning

counter = 0

dataframes_loc_list = []


# Attempting to iterate them
while counter < len(channels_available):
    try:
        skip_start_channel = position_list_channels[counter][0] + skip_start  
        skip_end_channel = skip_start + position_list_channels[counter + 1][0] + 1
    except:
        skip_start_channel = position_list_channels[counter][0] + skip_start 
        skip_end_channel = skip_start + last_end_time[0] + 4
        
    dataframe_channel_name = 'channel_{}'.format(channels_available[counter])

    dataframes_loc_list.append([skip_start_channel, skip_end_channel])
 
    counter += 1


# In[21]:


## Goal of the Chunk: Use the indices and extract the data for each channel iteratively

# Creating a blank dataframe to prevent from .append() for adding redundant information
master_list = pd.DataFrame()

# Storing the list of channel names to be added to the blank dataframe
master_list_template_spacers = {"": []}

# Creating a blank dataframe to prevent the information from a previous run from impacting future runs of this code
data_imported_channel = pd.DataFrame()

# Creating a dataframe based off the empty dictionary previously
master_list = pd.DataFrame(master_list_template_spacers)

# Creating an empty list to store values about how long each of the dataframes are
len_counter = []

# Storing if the data inside each dataframe need to be transposed or not
true_counter = 0

# Using counter to cycle through values in the channels_available list
counter = 0

# Iteratively storing dataframes per channel
for location in dataframes_loc_list:
    # Recording the start and end information in the overall document- preserving indexing from before
    skip_start_channel = location[0]
    skip_end_channel = location[1] - 1
    # Loading the file again, using the indexing associated with each channel
    data_imported_channel = pd.read_excel(input_file_path, header = skip_start_channel - 1)
    data_imported_channel = data_imported_channel[:(skip_end_channel - skip_start_channel - 2)]  
    # Inserting a Channel column that cycles through the channels_available list using the counter indexing
    data_imported_channel.insert(0, "Channel", channels_available[counter])
    # Readjusing the header
    data_imported_channel.columns = data_imported_channel.iloc[0]
    # Renaming the first column to be called "Channel"
    data_imported_channel = data_imported_channel.rename(columns={data_imported_channel.columns[0]: 'Channel'})
    # Deselects the first row that was used to make the header
    data_imported_channel = data_imported_channel[1:]
    # Appending each dataframe to the dataframe
    master_list = master_list.append(data_imported_channel)
    counter += 1 
    

    # Taking into account the number of Cycles- will help add Channel info for data that need to be transposed
    if true_counter == 0:
        len_counter.append(data_imported_channel.shape[1] - 3)

# Dropping the first column from the dataframe    
data_imported_processed = master_list.drop(master_list.columns[0], axis=1)


# In[22]:


## Goal of the Chunk: Either leave data in the correct form as they are or transposing data in the incorrect form
## Previous chunks did not use the inside of the data to preserve indexing

# Creating Empty lists / resetting counters 
channel_list = []
channels_to_be_added = []
counter = 0
true_counter = 0


# Re-establishing the true_counter from before
for col in data_imported_processed:
    if col == "Time [s]":
        # There should only be one "Time [s]"
        true_counter += 1
    else:
        pass


# Transposes data if they are in the incorrect format and leaves data as they are if in the correct format
if true_counter == 0:
    #  Renames the first column to be "Channel"
    data_imported_processed_t = data_imported_processed.rename(columns={data_imported_processed.columns[0]: 'Channel'})
    # Drops the first channel
    data_imported_processed_t = data_imported_processed_t.drop(data_imported_processed_t.columns[0], axis=1)
    # Transposes the data
    data_imported_processed_t = data_imported_processed_t.T
    # Resets the index after the column was dropped and takes out Cycle Nr. from being an index
    data_imported_processed_t = data_imported_processed_t.reset_index()
    # Makes the first row the header of the dataframe
    data_imported_processed_t.columns = data_imported_processed_t.iloc[0]
    # Ignores the first row now that it is the header of the dataframe
    data_imported_processed_t = data_imported_processed_t[1:]
    # Converts the Cycle Nr. column to be the integer type instead of a float with crazy decimals
    data_imported_processed_t["Cycle Nr."] = data_imported_processed_t["Cycle Nr."].astype(int)
    # Uses the len_counter information to append the channel name the number of times there are cycles
    for channel in channels_available:
        for i in range(len_counter[counter] + 1):
            channels_to_be_added.append(channel)
    # Creates a dataframe with the channels_to_be_added (making this the first row of the concat dataframe)
    channel_df = pd.DataFrame(channels_to_be_added)
    # Renames the column as "Channel"
    channel_df = channel_df.rename(columns={channel_df.columns[0]: 'Channel'})
    # The channel_df has an index that starts at 0 while the index for the channel dataframes start at 1
    # Adding 1 to the channel_df allows the concat to work properly
    channel_df.index = channel_df.index + 1
    # Combining the channel_df and the data_imported_processed_t via their now-shared index
    data_imported_processed = pd.concat([channel_df, data_imported_processed_t], axis=1)
else:
    # Leaves the data as they are, renaming the dataframe in the process
    pass


# In[23]:


## Goal of the Chunk: Remove empty columns of wells with no data and replace "OVER" values with NaNs. 
## Since empty wells are read in as NaNs, this would distinguish empty wells from "OVER" values,
## providing more directed feedback

# Creating Empty Lists/Dictionaries / Resetting Counters
list_of_wells_nans = []
list_of_wells = []
list_of_wells_nans_dict = {} 
filtered_dictionary = {}
counter = 0


# Ignoring these values and extracting the wells that were potentially used (they are columns)
for col in data_imported_processed:
    if col == "Channel":
        pass
    elif col == "Cycle Nr.":
        pass
    elif col == "Time [s]":
        pass
    elif col == "Temp. [°C]":
        pass
    else:
        list_of_wells_nans.append(pd.isna(data_imported_processed[col]))


# Transposing the data to use the information in wells as rows
data_imported_processed_wells_extraction = data_imported_processed.T.reset_index()

# pot_well (potential_well) looks through the index of the transposed data and looks for the wells
# Each well will appear multiple times depending on the channel(s) used
for pot_well in data_imported_processed_wells_extraction["index"]:
    if pot_well == "Channel" or pot_well == "Cycle Nr." or pot_well == "Time [s]" or pot_well == "Temp. [°C]":
        pass
    else:
        # Appends the wells to a list
        list_of_wells.append(pot_well)
    
        
# It is helpful to know how many missing values there are per well (row == False)
for well in list_of_wells_nans:
    for row in well:
        if row == False:
            if list_of_wells_nans_dict.get(list_of_wells[counter]) is None:
                # If the entry for a well is not in yet, it is added to the dictionary
                list_of_wells_nans_dict[list_of_wells[counter]] = 1
            else:
                # If an entry has already been added for a well, the counter of missingness increases for that entry
                list_of_wells_nans_dict[list_of_wells[counter]] += 1
        else:
            pass
    counter += 1 
            
# Cycles through the dictionary values and removes wells that are not used at all
for key, value in list_of_wells_nans_dict.items():
    if (value > len(data_imported_processed["Channel"])):
        filtered_dictionary[key] = value

# Resetting the list_of_wells values
list_of_wells = []

# Appends the keys of the well entries to a list
for key in list_of_wells_nans_dict:
    list_of_wells.append(key)
    
# Appended information gets added to the columns_to_keep- essential information of the dataframe
columns_to_keep = ["Channel", "Cycle Nr.", "Time [s]", "Temp. [°C]"] + list_of_wells
# Keeping only the columns desired
data_imported_processed = data_imported_processed[columns_to_keep]

# Changing "OVER" values to NaNs
data_imported_processed = data_imported_processed.replace({"OVER": "NaN"}, regex=True)


# # In[28]:
# Writing a Blank Excel file to ensure reruns don't add to it
file_complete = output_path
blank_slate = pd.DataFrame()
with pd.ExcelWriter(file_complete,
                    mode='w') as writer:  
    blank_slate.to_excel(writer, sheet_name='Data')
# This portion rewrites a second sheet, Wells, to the existing Excel file with the Data sheet
with pd.ExcelWriter(file_complete,
                    mode='w') as writer:  
    blank_slate.to_excel(writer, sheet_name='Wells')
    
# This creates an excel file with the Data sheet
data_imported_processed.to_excel(file_complete, sheet_name = "Data")

# Adapted from: https://www.javatpoint.com/how-to-create-a-dataframes-in-python
# Making a Template for Wells
wells_template_spacers = {"Well": [], "Condition": [], "Background_group": [], "Strain": [], "Group": [], "Control": []}


# Since wells_template_spacers is a list, the items in the list need to be added one by one
for well in list_of_wells:
    # The wells will get filled in for the user
    wells_template_spacers["Well"].append(well)
    # The user will unfortunately have to load in the information by themselves for the next columns,
    # but this at least sets everything else up for them
    # The blanks need to be created to ensure each row has at least something for each column
    # because how can a column have a row but nothing in it? -at least this is how pandas sees this problem
    wells_template_spacers["Condition"].append("")
    wells_template_spacers["Background_group"].append("")
    wells_template_spacers["Strain"].append("")
    wells_template_spacers["Group"].append("")
    wells_template_spacers["Control"].append("")
    
# Creating a new dataframe that will serve as the template for the Wells sheet
wells_template = pd.DataFrame(wells_template_spacers)

# This portion appends a second sheet, Wells, to the existing Excel file with the Data sheet
with pd.ExcelWriter(file_complete,
                    mode='a', engine="openpyxl") as writer:  
    wells_template.to_excel(writer, sheet_name='Wells', index = False)
