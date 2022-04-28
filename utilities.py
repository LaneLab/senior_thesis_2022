#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Goal of the Chunk: Importing the necessary package for the handle_input() function
import os


# In[3]:


## Goal of the Chunk: Handling a variety of different inputs
def handle_input(input_file_path, output_dir_path=""):
    # If the user supplies one argument, the argument_list should only contain one element.
    if output_dir_path == "":
        argument_list = [os.path.isfile(input_file_path)]
    # If the user supplies two arguments, the argument_list should have two elements in which both are checked.
    else:
        argument_list = [os.path.isfile(input_file_path), os.path.isdir(output_dir_path)]

    Found = False


    ## This is borrowed from the combining_notebooks_interactive.ipynb
    # Option 1
    if len(argument_list) == 2:
        # Both the input file name is valid and the output directory exists
        if os.path.isfile(input_file_path) == True and os.path.isdir(output_dir_path) == True:
            # Splits the file name, called input_file_name, from the rest of the path
            input_dir, input_file_name = os.path.split(input_file_path)
            # places the word "processed_" before the original file name to indicate the data have been modified.
            output_path = output_dir_path
            Found = True
        # The input file name is valid but the output directory is yet to exist
        elif os.path.isfile(input_file_path) == True and os.path.isdir(output_dir_path) == False:
            output_dir_check_above, output_dir_check = os.path.split(output_dir_path)
            # Taking everything above where the new directory is created and separating from the folder that will be created
            if os.path.isdir(output_dir_check_above) == True:
                input_dir, input_file_name = os.path.split(input_file_path)
                # Make a new directory and storing the file there with previously spliced information
                new_dir = os.path.join(output_dir_check_above, output_dir_check)
                os.mkdir(new_dir)
                # Saving the file in the new directory
                output_path = new_dir
                Found = True
            # Handles an incorrect path above where the new directory wants to be made
            else:
                print("The directory supplied does not have a valid path")
                pass
        # Handles incorrect input file- completely incorrect path supplied by at least the input, possibly the output too
        else:
            print("Either the input file name or the output directory name was entered incorrectly.")
    # Option 2
    elif len(argument_list) == 1:
        # Saving the output file in the same directory as the valid input file provided
        if os.path.isfile(input_file_path) == True:
            input_dir, input_file_name = os.path.split(input_file_path)
            output_path = input_dir
            Found = True
        # Handles an invalid input
        else:
            print("Error: The input file name was entered incorrectly.")
    # Too many arguments
    elif len(argument_list) > 2:
        print("Error: Too many arguments given. There should be a maximum of two.")
    # Not enough arguments given
    else:
        print("Error: Not enough argument(s) given.")
        print("There should be at least one correct file path with the option of a new directory given.") 
    
    plot_path_to_save = output_path
    
    return Found, plot_path_to_save


# In[18]:


## Goal of the Chunk: Defining a Function with default parameters

# Defining functions using def function_name(parameter:type=default_value): is very helpful
# This allows all the steps performed within the function to be done every time the user types scaling_params()
def scaling_params(total_time:int, x_range_min:int=0, x_range_max:int=None):
    
    # Notes on the arguments / internal variables ----------------------
    # The total_time argument is a required argument where the endpt variable is passed in the plotting script.
    # This takes into account how long data were collected for.
    
    # The x_range_min is an optional argument that has a default value of 0hours.
    # If the user wants to specify at which hour to start plotting, they may provide an input
    
    # The x_range_max is an optional argument that will essentially override the value stored in the total_time argument.
    # If the user does not want to plot the entirety of the data, they may input at which hour they would like to stop.
    
    # This function uses two more variables to make its calculations: min_value and max_value.
    # These values refer to the minimum and the maximum values of the x-axis.
    # ------------------------------------------------------------------
    
    # This handles the different inputs.
    if not x_range_max:
        # If the user does not supply any additional arguments besides the value for total_time,
        # the default is to set the max_value as total_time and plot all the data
        max_value = total_time
    else:
        # If the user supplies an additional optional argument for x_range_min, that will
        # be the value stored for max_value
        max_value = x_range_max
    
    # min_value is either the default value specified in the function, 0, or whatever the user wants it to be
    # using the x_range_min optional argument.
    min_value = x_range_min
    
    # It does not make sense to plot from 0hours to 0hours, so that is taken care of.
    if max_value == 0:
        print("Error: The maximum value cannot be 0 hours.")
    
    elif max_value - min_value <= 0:
        # Checking to make sure the input makes sense
        # it does not make sense to plot from 0hours to 0hours
        # and it does not make sense to plot from 4hours to 2hours- at least not for these kinds of problems
        print("Error: The maximum value should be larger than the minimum value.")
    
    
    else:
        # Setting up blank lists to erase the information from the last time the user used this function
        hours_list = []
        breaks_list = []
        
        
        
        # This will determine how many breaks and how far apart they should be
        if max_value - min_value > 0 and max_value - min_value < 2:
            # When the measurements are one hour or less, make scale every 6 minutes and up to an hour
            for i in range(12):
                hours_list.append(min_value + round(i * 0.10, 2))
                breaks_list.append(round(i * 6, 2))
        elif max_value - min_value < 3:
            # When the measurements are between two (inclusive) and three hours (exclusive), make scale every 15 minutes
            for i in range(10):
                hours_list.append(min_value + round(i * 0.25, 2))
                breaks_list.append(round(i * 15, 2))
        elif max_value - min_value == 3:
            # When the measurements are three hours only, make scale every 20 minutes
            for i in range(11):
                hours_list.append(min_value + round(i * 1/3, 2))
                breaks_list.append(round(i * 20, 2))
        elif max_value - min_value > 3 and max_value - min_value < 5:
            # When the measurements are between three (exclusive) and five hours (exclusive), make scale every 30 minutes
            for i in range(10):
                hours_list.append(min_value + round(i * 0.5, 2))
                breaks_list.append(round(i * 30, 2))
        elif max_value - min_value >= 5 and max_value - min_value < 8:
            # When the measurements are between five (inclusive) and 8 hours (exclusive), make scale every 60 minutes, aka 1 hour
            for i in range(max_value + 1):
                hours_list.append(min_value + i)
                breaks_list.append(round(i * 60, 2))
        else:
            # When the measurements are greater than 8 hours, make scale every two hours
            for i in range(max_value + 1):
                if i % 2 == 0:
                    hours_list.append(min_value + i)
                    breaks_list.append(round(i * 60, 2))


        # Resetting from any previous calls of this function
        # This needs to be done because values are later appended to the list.
        # Without resetting labels_list, the list would continue to grow with each time, which is a big problem.
        labels_list = []


        # Adjusting for nicer labels
        # This is the part that displays the time on the x-axis as hour:minute
        label_counter = 0
        for b in breaks_list:
            # The hours_list is in the form of the hour. The int() ensures that it is recorded as an integer.
            # breaks_list is a list of the time in minutes
            # Finding the remainder of the minutes divided by 60 says how many minutes have passed.
            # For example, 30 / 60 has a remainder of 30 (true for all minutes), 
            # while 60 / 60 has a remainder of 0 (true for all hours)
            labels_list.append(str(int(hours_list[label_counter])) + ":" + str(b % 60))
            label_counter += 1


        # The 'scale_x_continuous()' function of 'ggplot()' needs to know the limits of the x-axis
        # It is easiest to store these values as a list to shorten the code involved within the 'ggplot()'
        limits_list = [breaks_list[0], breaks_list[-1:][0]]

        # It would be useful to know which hours were plotted.
        # This would prevent plots from 0 to 8 hours from being overwritten by plots from 3 to 6 hours.
        hours_to_filename = str(min_value) + "_" + str(max_value)


        # Using 'return' is very important to functions. 
        # These values will be returned as a list.
        # Using indexing, I can refer to these values by using 'scaling_params(min_value, max_value)[index]'.
        # This will make the code much more flexible and make plots customizeable each time.
        return breaks_list, hours_list, min_value, max_value, labels_list, limits_list, hours_to_filename
    #                0           1         2          3             4            5           6  

