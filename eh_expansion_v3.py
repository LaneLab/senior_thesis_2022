## Loading Packages
import os
import numpy as np
import pandas as pd
from math import floor
from scipy.interpolate import interp1d
from openpyxl import *

user_mods = False




def opt_in_detection(overall_path):
    
    '''
    Overview / Goal:
    The overall goal of this function is to assess whether or not the 
    user wants to make additional calculations. This opt-in approach 
    will need to follow my specified rules of having an opt_in directory 
    with an Excel sheet in the correct format. As long as the user follows 
    those restrictions, they should be able to opt-in to additional calculations.

    Arguments:
    * overall_path: This takes the directory of where the *_summary.txt 
    output will be stored. It will then truncate the \\summary portion 
    (for Windows) of the path. From there, it will add on \\opt_in and 
    see if the folder exists as well if there is an Excel file.
    Case I:
    If both of those components exist, then it will indicate that the user 
    requested additional calculations (user_mods == True) as well as the read 
    in Excel file (parameters_xlsx).
    Case II:
    If Neither of these components exist, the user has indicated that they do 
    not want additional calculations (user_mods == False) and that there is no 
    excel file (0, serving as a placeholder to mimic the dimensions of Case I)
    '''

    global user_mods 
    # Figuring out where the user is located.
    # This is assumed to be using a path that looks at all AMiGA folders: data, derived, figures...
    # base_path is summ_path but without \\summary at the end
    provided_summ_path, summ = os.path.split(overall_path)
    base_path, summ = os.path.split(provided_summ_path)
    opt_in_dir = os.path.join(base_path, "opt_in") 
    
    # If there is nothing in the opt_in_dir (e.g., directory does not exist),
    # it will produce an error when I try to subset it. 
    try:
        # When the User wants to Opt In
        parameters_file_name = os.listdir(opt_in_dir)[0]
        opt_in_path = os.path.join(opt_in_dir, parameters_file_name)
        parameters_xlsx = pd.read_excel(opt_in_path, engine = 'openpyxl')
        user_mods = True
        return user_mods, parameters_xlsx
    except:
        # When the User wants to continue with the defaults
        user_mods = False
        # Using a placeholder of 0 to preserve the dimensions (no DataFrame in this place)
        return user_mods, 0




def duration_extractor(overall_path):  

    '''
    Overview / Goal:
    The goal for this function is to take in a path of the original 
    data and determine what the duration of the experiment should be 
    to make sure that everything loaded correctly. 

    Argument:
    * overall_path: This path takes advantage of the summ_path in the 
    AMiGA analysis.py file. This then points this function to the 
    correct file location to carry out its calculations.
    '''
    
    # This is taking advantage of an existing file storage structure to look for things
    provided_summ_path, summ = os.path.split(overall_path)
    base_dir, summ_dir_name = os.path.split(provided_summ_path)
    processed_input_data_dir = os.path.join(base_dir, "data")
    
    # This finds the file with the .txt and uses that as an input
    os_list_dir = os.listdir(processed_input_data_dir)
    for item in os_list_dir:
        if item[-4:] == ".txt":
            item_of_interest = item

    processed_input_txt_path = os.path.join(processed_input_data_dir, item_of_interest)

    # print(processed_input_xlsx_path)
    data = pd.read_csv(processed_input_txt_path, sep = "\t")

    # Defining an empty list to reset
    list_times_exp = []

    # The times are listed as columns, which is what this for loop extracts.
    # Most of the time, the times aren't exactly 900 seconds, but '901.29283' seconds
    for time in data:
        print(time)
        try:
            time = float(time)
            time = floor(time)
            list_times_exp.append(time)
        except:
            pass
            

    # This then looks at the largest time
    last_obs = max(list_times_exp)
    # The plate reader is in seconds, AMiGA works in hours
    time_hours = last_obs / 3600
    # This takes that estimate and rounds it to the nearest 2 decimal places.
    # It would be very strange if someone ran an experiment for an irrational duration.
    time_hours = round(time_hours, 2)
    # This then is stored as the duration, which is referenced in other places
    duration = time_hours
    return duration




def interval_extractor(overall_path):
    
    '''
    Overview / Goal:
    I unfortunately cannot have this function change the value in the config.py
    file since that would be a circular import. However, I still find this useful
    to make sure that the user supplied the correct value to AMiGA for the interval.
    This function helped me out when I had an interval in the config.py file that did
    not match the interval of my experiment.

    Argument:
    * overall_path: Again, this is taking advantage of the existing AMiGA framework.
    This function will take information about the duration and then use that in context
    of the experiment to determine the length of the interval between measurements.
    '''

    # Reading in the data
    provided_summ_path, summ = os.path.split(overall_path)
    base_dir, summ_dir_name = os.path.split(provided_summ_path)
    un_processed_input_data_dir = os.path.join(base_dir, "experiment_data")
    un_processed_input_xlsx = os.listdir(un_processed_input_data_dir)[0]
    un_processed_input_xlsx_path = os.path.join(un_processed_input_data_dir, un_processed_input_xlsx)
    ####
    # Removing the amiga_processed_processed portion
    summ_trunc = summ[26:]
    # Removing the _summary.txt portion
    summ_trunc = summ_trunc[:-12]
    un_processed_input_xlsx_path = os.path.join(un_processed_input_data_dir, summ_trunc)
    # Appending the .xlsx portion.
    # AKA, I'm assuming that the user will supply an Excel file.
    # I always work in Excel, so it doesn't make sense for me to anticiapte deviations
    un_processed_input_xlsx_path += ".xlsx"
    ####
    df_to_count = pd.read_excel(un_processed_input_xlsx_path, engine = 'openpyxl')
    
    # Resetting this value each time
    row_counter = 0

    # This for loop counts the number of times appearing in the 'experiment_data' folder
    for row in df_to_count["Time [s]"]:
        float(row)
        row_counter += 1
    # There are n-1 intervals for n timepoints 
    # (1,2 has 1 interval; 1,2,3 has 2 intervals....)
    config_value = duration_extractor(overall_path) * 3600 / (row_counter - 1)
    # This rounds what the config_value should be to 2 decimal places
    config_value = round(config_value, 2)
    # To avoid instances where the times are slightly off (900 versus 901 seconds),
    # this excludes the remainder and then readjusts the value
    config_value = config_value // 100 * 100
    return config_value




def redundancy_checker(x_i, y_i, list_times_to_intp1d, list_ods_to_intp1d, exp_duration:float):
    
    '''
    Overview / Goal:
    As the name suggests, this function takes in single values and checks to see if
    they are present in the list (my cleaning function will remove them if they are
    both present). The single values (x_i and y_i) are checked against their respective
    lists. At the very end, I take advantage of the properties of a Python `set()`, and
    that automatically removes duplicate items in my lists. That means that any redundant 
    values are successfully removed. The redundancy checker also removes time values 
    that exceed that of the duration of the experiment to prevent errors in future 
    steps. This function exists within the arg_extractor class to catch redundancy 
    before it is used in other locations.

    Arguments:
    * x_i: This is the single time to interpolate for supplied by the user.
    * y_i: This is the single OD-value to interpolate for supplied by the user
    * list_times_to_intp1d: This is a list of times to interpolate for that are supplied
    for by the user. Each time is checked to see if it is less than the duration of the
    experiment (I don't want my function to interpolate beyond the scope of the experiment).
    Any values that are the same as x_i are removed.
    * list_ods_to_intp1d: This makes sure that no values identical to y_i are included.
    * exp_duration: This is important to making sure every time makes sense to interpolate
    for (all times are checked against this value). 
    '''

    # Resetting lists
    list_times_to_intp1d_updated = []
    list_ods_to_intp1d_updated = []
    duration = exp_duration
    
    # This for loop first checks that the x-value is not the same as the x_i supplied
    for time in list_times_to_intp1d:
        if time == x_i:
            pass
        else:
            if time <= duration:
                list_times_to_intp1d_updated.append(time)
            else:
                pass
    
    # This makes sure that x_i itself does not exceed the duration of the experiment
    # In that case, it becomes None, which then doesn't get interpolated for later on.
    if x_i >= duration:
        x_i = None

    # This for loop checks to make sure that no OD-values are in the 
    for od_value in list_ods_to_intp1d:
        if od_value == y_i:
            pass
        else:
            list_ods_to_intp1d_updated.append(od_value)

    # set() removes duplicates by definition.
    # I then want to turn it back to a list to stay consistent with other portions.
    # All in all, this removes duplicates from within the list.
    list_times_to_intp1d = list(set(list_times_to_intp1d_updated))
    list_ods_to_intp1d = list(set(list_ods_to_intp1d_updated))
    
    return x_i, list_times_to_intp1d, list_ods_to_intp1d



def sign_tracker(sign_list:list):
    
    '''
    Overview / Goal:
    This function is a part of the arbitrary_od_finder_od() function.
    It takes the sign_list from there and counts how many times per sample
    they pass. If there are multiple crossings, that is noted by the 
    sign counter.

    Argument:
    * sign_list: This is a list of signs for each OD interpolation for each sample.
    This then keeps track of the number of times an curve crosses the interpolated
    OD-value.
    '''

    # Resetting value
    sign_change_counter = 0

    # This for loop looks starts at the second sign and looks back
    # to the sign before it and sees if they are different.
    # e.g., -+ results in sign_change_counter += 1
    # while -- results in a pass
    for i in range(1, len(sign_list)):
        if sign_list[i-1] == sign_list[i]:
            pass
        else:
            sign_change_counter += 1
    
    return sign_change_counter



def sample_id_well_conversion(well_of_interest:str):
    
    '''
    Overview / Goal:
    This function is clunky, but it changes the Sample_ID number (I view
    this as more of the numeric form) into a Well identification in the 
    format RowColumn (I view this as the graphical form that is easy to 
    envision on the plate) and returns that value (e.g., 0 is equivalent to A1).
    This comes in handy for the arbitrary_od_finder_od() function, specificially
    helping the user understand which wells should be looked at more closely.

    Argument:
    * well_of_interest: This takes in an integer and returns the well in the format
    of RowColumn. Based on the dimensions of a 96 well plate, the domain of this
    should only ever be [0, 95]. 
    '''


    # First Creating a List of all 96 wells
    well_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    well_list = []

    # This makes the wells go from A1-H12 without me 
    # having to manually type out a list of all 96 wells
    for well_letter in well_letters:
        for i in range(1, 13):
            well_list.append(well_letter + str(i))


    # Erasing Previous iterations' work
    sample_counter = 0
    id_well_dict = {}
    Found = False

    # This makes a dictionary that takes in sample_counter
    # as the key (Sample_ID) and maps it to the value, well
    # This increments the well and the Sample_ID by 1 place 
    # for each iteration
    for well in well_list:
        id_well_dict[sample_counter] = well
        sample_counter += 1
            
    
    # This for loop then stores the well of interest in the 
    # geographic form rather than its entered numerical form
    for sample_id, well in id_well_dict.items():
        if sample_id == well_of_interest:
            Found = True
            return_value = well
        # This is here in the event that the data are entered
        # very strangely
        elif Found == False:
            message = "The well was inputted incorrectly."
            return_value = message

    return return_value 
    



def arbitrary_od_finder_time(input_gp_data, interp1d_kind:str, arbitrary_x:float, exp_duration:float):
    
    '''
    Overview / Goal:
    To take a DataFrame of AMiGA derived values for all wells and interpolate 
    OD observations between observed points for all samples. This function 
    uses the Scipy Interp1d (Interpolating 1D arrays) function to estimate the 
    values. From there, I add on a weighted average based on the x distance 
    between the two closest points for a given time. The distance-based weighted 
    average of the two closest points for a given time is then returned for all 
    values (master_df_limited). The other option is to print out all 
    interpolated values for every sample (master_df_full). Due to the size 
    (and I have not thought about how to store this), this is not yet stored, 
    but would be an additional feature.

    Arguments:
    * input_gp_data: This uses the sub_plate.data attribute to access the 
    gaussian process data (gp_data).
    * interp1d_kind: This has to do with the scipy package and its method 
    of 1D interpolation. Some common options are linear or cubic.
    * arbitrary_x: This is the time where the value is to be interpolated at. 
    This could be a random HOUR, or a lag time (hopefully this feature is 
    followed up on). 
    * exp_duration: This prevents time values larger than the experiment 
    from being interpolated.
    '''
    
    ## Defining Chunks to be Reset (Outside Loop)
    # Handling Arguments
    gp_info = input_gp_data
    x_i = float(arbitrary_x)
    interp1d_kind = interp1d_kind
    duration_input = exp_duration
    

    # Handling Other Intermediate Steps
    df_list = []
    time_list = []
    
    interpolated_name = "Interpolated_Time_" + str(x_i)

    
    # I want the master_df to have the same column names, but no filler values included in the final return
    master_df_limited = pd.DataFrame({"Sample_ID":[0], "Time":[0], interpolated_name:[0]})
    master_df_limited = master_df_limited[1:]
    
    #  Sample_ID    xnew  f_gp(xnew)
    master_df_full = pd.DataFrame({"Sample_ID":[0], "xnew":[0], "f_gp(xnew)":[0]})
    master_df_full = master_df_full[1:]
    
    if x_i <= duration_input:
        ## Creating a List of DataFrames based on their Sample_ID
        ## Going from 0 to 95, this is covered by the range(96) and adds the DataFrames to a list that I can
        ## then iterate through a for loop (next loop)
        for i in range(96):
            df_list.append(gp_info[gp_info.Sample_ID == i])

            
        ## Looping Through Each DataFrame in the List
        for id_num in range(96):
            ## Defining Chunks to be Reset (Inside Loop)
            sample_id_filler = []
            time_dict = {}
            time_counter = 0
            sign_list = []
            counter = 0 
            nan_truth = False            
            
            ## The part that selects the one DataFrame to work through at a time        
            data_to_use = df_list[id_num]
            
            
            ## Automating the process of determining how long each experiment is run for
            for time in data_to_use["Time"]:
                # time = float(time)
                time_list.append(time)
            # Finding the maximum value in the stored list- will be helpful in the np.linspace() function
            duration = max(time_list)
            
            ## Interpolating the od values
            ## Following the scipy documentation:
            ## https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
            x_gp = data_to_use["Time"].to_list()
            y_gp = data_to_use["OD_Fit"].to_list()   
            f_gp = interp1d(x_gp, y_gp)
            f2_gp = interp1d(x_gp, y_gp, kind = interp1d_kind)
            #****** Arbitrary Alert ******#
            # Not only is the number of points to interpolate arbitrary,
            # but also the start and endpoint could be altered.
            # For the purpose of simplicity, I will assume that the user wants to
            # potentially interpolate over the whole experiment.
            xnew = np.linspace(0, duration, num=250)
            #*****************************#
            
            # Creating a Primary Key using Sample_ID as a list
            frame_length = len(xnew)
            for i in range(frame_length):
                sample_id_filler.append(id_num)
                
                
            ## Storing interpolated values in a DataFrame
            ## This will be searched and values compared against the input arbitrary_x
            f_xnew_df_gp = pd.DataFrame(f_gp(xnew)).rename(columns = {0:"f_gp(xnew)"})
            xnew_df_gp = pd.DataFrame(xnew).rename(columns = {0:"xnew"})
            sample_id_gp = pd.DataFrame(sample_id_filler).rename(columns = {0:"Sample_ID"})
            input_df = pd.concat([sample_id_gp, xnew_df_gp, f_xnew_df_gp], axis = 1)   
            master_df_full = master_df_full.append(input_df)
        
            ## Looking at the "Times" column and finding the closest distance to a point, arbitrary_x
            for time in input_df["xnew"]:
                # time = float(time)
                difference = pow(pow((time - x_i), 2), 0.5)
                # Keeping track if the point of interest is above or below observed points
                sign_indicator = x_i - time
                sign_list.append(sign_indicator)
                # Adding the key and value to the dictionary 
                if time_counter in time_dict:
                    pass
                else:
                    time_dict[time_counter] = difference
                time_counter += 1



            # Documenting the minimum value for distance- this is the closest point
            time_dict_dist_min = min(time_dict.values())
            # Locating the time_counter for the minimum distance and saving its index as time_index
            for time_counter, distance in time_dict.items():
                if distance == time_dict_dist_min:
                    time_index = time_counter


                    
            ## Splicing the interpolated DataFrame to only include one point above and below the arbitrary_x point
            # If the value is below its closest point, the script should grab the row below the closest observed point
            # if sign_list[time_index] < 0 and input_df["f_gp(xnew)"][time_index] > 0:
            if sign_list[time_index] < 0:
                min_distance_frame = input_df[(time_index-1):(time_index + 1)]
            # If the value is above the closest point, then the script should grab the row above the closest observed point
            # elif sign_list[time_index] > 0 and input_df["f_gp(xnew)"][time_index] > 0:
            # elif sign_list[time_index] > 0:
            else:
                min_distance_frame = input_df[time_index:(time_index + 2)]
            
            # Storing the values closest to the arbitrary_x in a more useable format
            for time, pred_od in zip(min_distance_frame["xnew"], min_distance_frame["f_gp(xnew)"]):
                if counter == 0:
                    x_a = time
                    y_a = pred_od
                elif counter == 1:
                    x_b = time
                    y_b = pred_od
                else:
                    pass
                counter += 1
            
            
            ## Calculating the distance in preparation for a weighted average
            if nan_truth == False:
                da = pow(pow((x_i - x_a), 2), 0.5)
                db = pow(pow((x_i - x_b), 2), 0.5)
                dt = da + db
                y_w_avg = (db/dt)*y_a + (da/dt)*y_b
            elif nan_truth == True:
                y_w_avg = "Nan"

            
            ## Storing the values and iteratively appending the values to a master_df that will hold
            ## an interpolated OD value for all samples at the arbitrary_x point
            inter_df_min = pd.DataFrame({"Sample_ID":[id_num], "Time":[x_i], interpolated_name:[y_w_avg]})
            master_df_limited = master_df_limited.append(inter_df_min)
        
        master_df_limited_time = master_df_limited
        master_df_full_time = master_df_full
        return master_df_limited_time, master_df_full_time     

    else:
        return None, None


def arbitrary_od_finder_time_list(input_gp_data, summary_data, provided_interp1d_kind:str, provided_list_times_to_intp1d:list(), exp_duration:float):

    '''
    Overview / Goal:
    A way to interpolate OD Values for a list of Times provided 
    (automating arbitrary_od_finder_time()).

    Arguments:
    * input_gp_data: This uses the sub_plate.data attribute to access 
    the gaussian process data (gp_data).
    * summary_data: This uses the sub_plate.key attribute to access the 
    DataFrame that will be used in the *_summary.txt output.
    * provided_interp1d_kind: This has to do with the scipy package and 
    its method of 1D interpolation. 
    * provided_list_times_to_interp1d: Similar to the arbitrary_od_finder_time(), 
    but this time a list of times to interpolate for is provided.
    * exp_duration: The total time that the experiment is run for. 
    This prevents an invalid input from being interpolated.
    '''

    # Reading in arguments
    gp_data = input_gp_data
    summary_df = summary_data    
    list_times_to_intp1d = provided_list_times_to_intp1d
    interp1d_kind = provided_interp1d_kind
    duration_input = exp_duration

    # This is all wrapped in an if statement,
    # but ensuring that these calculations are only run
    # when the user wants them to be
    if user_mods == True: 
        fuller_df = pd.DataFrame()
        intp1d_counter = 0
        input_gp_data = gp_data
        summary_df = summary_data


        # This adds DataFrames iteratively.
        # Adding the DataFrame n works differently than
        # DataFrame n + i
        for time_intp1d in list_times_to_intp1d:
            if intp1d_counter == 0:
                min_od_df = arbitrary_od_finder_time(gp_data, interp1d_kind, time_intp1d, duration_input)[0]
                fuller_df = pd.merge(min_od_df, summary_df, on=("Sample_ID"), how = "inner")
                newest_df = fuller_df
            else:
                new_od_df = arbitrary_od_finder_time(gp_data, interp1d_kind, time_intp1d, duration_input)[0]
                newest_df = pd.merge(new_od_df, newest_df, on=("Sample_ID"), how = "inner")
            intp1d_counter += 1
    # It shouldn't come to this else statement,
    # but just in case
    else: 
        newest_df = None
    
    time_list_intp1d_df = newest_df

    return time_list_intp1d_df




def arbitrary_od_finder_od(input_gp_data, interp1d_kind, arbitrary_y:float):
    
    '''
    Overview / Goal:
    To take a DataFrame of AMiGA derived values for all wells and interpolate 
    Time observations between observed points for all samples. This function 
    uses the Scipy Interp1d (Interpolating 1D arrays) function to estimate the 
    values. From there, I add on a weighted average based on the x distance 
    between the two closest points for a given OD Value. The distance-based 
    weighted average of the two closest points for a given OD Value is then 
    returned for all values (master_df_limited). The other option is to print 
    out all interpolated values for every sample (master_df_full). Due to the size 
    (and I have not thought about how to store this), this is not yet stored, 
    but would be an additional feature.

    Arguments:
    * input_gp_data: This uses the sub_plate.data attribute to access the 
    gaussian process data (gp_data).
    * interp1d_kind: This has to do with the scipy package and its method 
    of 1D interpolation. Some common options are linear or cubic.
    * arbitrary_y: This is the OD Value where the value is to be interpolated 
    at. This could be a random OD Value, or corresponding to a lag time 
    (hopefully this feature is followed up on). 
    '''
        
    ## Defining Chunks to be Reset (Outside Loop)
    # Handling Arguments
    gp_info = input_gp_data
    y_i = arbitrary_y
    interp1d_kind = interp1d_kind

    # Handling Other Intermediate Steps
    df_list = []
    time_list = []
    multiple_crossings = []
    
    interpolated_name = "Interpolated_OD_" + str(y_i)
    
    # I want the master_df to have the same column names, but no filler values included in the final return
    master_df_limited = pd.DataFrame({"Sample_ID":[0], "Time":[0], interpolated_name:[0]})
    master_df_limited = master_df_limited[1:]
    
    #  Sample_ID    xnew  f_gp(xnew)
    master_df_full = pd.DataFrame({"Sample_ID":[0], "xnew":[0], "f_gp(xnew)":[0]})
    master_df_full = master_df_full[1:]
    
    
    ## Creating a List of DataFrames based on their Sample_ID
    ## Going from 0 to 95, this is covered by the range(96) and adds the DataFrames to a list that I can
    ## then iterate through a for loop (next loop)
    for i in range(96):
        df_list.append(gp_info[gp_info.Sample_ID == i])

        
    ## Looping Through Each DataFrame in the List
    for id_num in range(96):
        ## Defining Chunks to be Reset (Inside Loop)
        sample_id_filler = []
        time_dict = {}
        time_counter = 0
        sign_list = []
        counter = 0 
        nan_truth = False
        sign_list_abstract = []
        
        
        ## The part that selects the one DataFrame to work through at a time        
        data_to_use = df_list[id_num]
        
        
        ## Automating the process of determining how long each experiment is run for
        for time in data_to_use["Time"]:
            time_list.append(time)
        # Finding the maximum value in the stored list- will be helpful in the np.linspace() function
        duration = max(time_list)
        
        
        
        ## Interpolating the od values
        ## Following the scipy documentation:
        ## https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
        x_gp = data_to_use["Time"].to_list()
        y_gp = data_to_use["OD_Fit"].to_list()   
        f_gp = interp1d(x_gp, y_gp)
        f2_gp = interp1d(x_gp, y_gp, kind = interp1d_kind)
        #****** Arbitrary Alert ******#
        # Not only is the number of points to interpolate arbitrary,
        # but also the start and endpoint could be altered.
        # For the purpose of simplicity, I will assume that the user wants to
        # potentially interpolate over the whole experiment.
        xnew = np.linspace(0, duration, num=250, endpoint=True)
        #*****************************#
        
        # Creating a Primary Key using Sample_ID as a list
        frame_length = len(xnew)
        for i in range(frame_length):
            sample_id_filler.append(id_num)
            
            
        ## Storing interpolated values in a DataFrame
        ## This will be searched and values compared against the input arbitrary_x
        f_xnew_df_gp = pd.DataFrame(f_gp(xnew)).rename(columns = {0:"f_gp(xnew)"})
        xnew_df_gp = pd.DataFrame(xnew).rename(columns = {0:"xnew"})
        sample_id_gp = pd.DataFrame(sample_id_filler).rename(columns = {0:"Sample_ID"})
        input_df = pd.concat([sample_id_gp, xnew_df_gp, f_xnew_df_gp], axis = 1)   
        master_df_full = master_df_full.append(input_df)
        
        ## Looking at the Interpolated Values column and finding the closest distance to a point, arbitrary_y
        for od_gp in input_df["f_gp(xnew)"]:
            od_gp = float(od_gp)
            y_i = float(y_i)
            difference = pow(pow((od_gp - y_i), 2), 0.5)
            # Keeping track if the point of interest is above or below observed points
            sign_indicator = y_i - od_gp
            sign_list.append(sign_indicator)

            if float(sign_indicator) < 0:
                sign_list_abstract.append("+")
            else:
                sign_list_abstract.append("-")

            # Adding the key and value to the dictionary 
            if time_counter in time_dict:
                pass
            else:
                time_dict[time_counter] = difference
            time_counter += 1

            
        # Counting the Number of crossings
        crossing_count = sign_tracker(sign_list_abstract)
        if crossing_count > 1:
            converted_well = sample_id_well_conversion(id_num)
            multiple_crossings.append(converted_well)
        else:
            pass

        # Documenting the minimum value for distance- this is the closest point
        time_dict_dist_min = min(time_dict.values())

        # Locating the time_counter for the minimum distance and saving its index as time_index
        for time_counter, distance in time_dict.items():
            if distance == time_dict_dist_min:
                time_index = time_counter

                
                
        ## This is a step particular to this function. 
        ## There will be instances where an OD of interest is not reached.
        ## My code will return Nans in those situations
        max_od_achieved = max(data_to_use["OD_Fit"])

        # Checking if the plate reader observed a sample that crossed the od of interest value
        if max_od_achieved < y_i:
            nan_truth = True
            
            
        ## Splicing the interpolated DataFrame to only include one point above and below the arbitrary_y point
        # If the value is below its closest point, the script should grab the row below the closest observed point
        # if sign_list[time_index] < 0 and input_df["xnew"][time_index] > 0:
        if sign_list[time_index] < 0:
            min_distance_frame = input_df[(time_index-1):(time_index + 1)]
        # If the value is above the closest point, then the script should grab the row above the closest observed point
        # elif sign_list[time_index] > 0 and input_df["xnew"][time_index] > 0:
        else:
            min_distance_frame = input_df[time_index:(time_index + 2)]
                        
        
        # Storing the values closest to the arbitrary_x in a more useable format
        for time, pred_od in zip(min_distance_frame["xnew"], min_distance_frame["f_gp(xnew)"]):
            if counter == 0:
                x_a = time
                y_a = pred_od
            elif counter == 1:
                x_b = time
                y_b = pred_od
            else:
                pass
            counter += 1
        
        
        ## Calculating the distance in preparation for a weighted average
        if nan_truth == False:
            da = pow(pow((y_i - y_a), 2), 0.5)
            db = pow(pow((y_i - y_b), 2), 0.5)
            dt = da + db
            y_w_avg = (db/dt)*x_a + (da/dt)*x_b
        elif nan_truth == True:
            y_w_avg = "Nan"

        
        ## Storing the values and iteratively appending the values to a master_df that will hold
        ## an interpolated OD value for all samples at the arbitrary_x point
        inter_df_min = pd.DataFrame({"Sample_ID":[id_num], "Time":[y_i], interpolated_name:[y_w_avg]})
        master_df_limited = master_df_limited.append(inter_df_min)
    
    master_df_limited_od = master_df_limited
    master_df_full_od = master_df_full

    return master_df_limited_od, master_df_full_od, multiple_crossings




def arbitrary_od_finder_od_list(input_gp_data, summary_data, provided_interp1d_kind:str, provided_list_ods_to_intp1d:list()):

    '''
    Overview / Goal:
    A way to interpolate OD Values for a list of Times provided 
    (automating arbitrary_od_finder_time()).

    Arguments:
    * input_gp_data: This uses the sub_plate.data attribute to access 
    the gaussian process data (gp_data).
    * summary_data: This uses the sub_plate.key attribute to access the 
    DataFrame that will be used in the *_summary.txt output.
    * provided_interp1d_kind: This has to do with the scipy package 
    and its method of 1D interpolation. 
    * provided_list_ods_to_interp1d: Similar to the arbitrary_od_finder_od(), 
    but this time a list of OD Values to interpolate for is provided.
    '''
    
    # Loading in arguments
    gp_data = input_gp_data
    summary_df = summary_data
    interp1d_kind = provided_interp1d_kind
    fuller_df = pd.DataFrame()
    intp1d_counter = 0
    list_ods_to_intp1d = provided_list_ods_to_intp1d

    # This adds DataFrames iteratively.
    # Adding the DataFrame n works differently than
    # DataFrame n + i
    for od_intp1d in list_ods_to_intp1d:
        if intp1d_counter == 0:
            min_od_df = arbitrary_od_finder_od(gp_data, interp1d_kind, od_intp1d)[0]
            fuller_df = pd.merge(min_od_df, summary_df, on=("Sample_ID"), how = "inner")
            newest_df = fuller_df
        else:
            new_od_df = arbitrary_od_finder_od(gp_data, interp1d_kind, od_intp1d)[0]
            newest_df = pd.merge(new_od_df, newest_df, on=("Sample_ID"), how = "inner")
        intp1d_counter += 1

    od_list_intp1d_df = newest_df

    ## Handling instances of multiple crossings
    multiple_xings_list = []
    multiple_xings_tf = False

    for od_intp1d in list_ods_to_intp1d:
        multiple_xings = arbitrary_od_finder_od(gp_data, interp1d_kind, od_intp1d)[2]
        if len(multiple_xings) != 0:
            for well in multiple_xings:
                multiple_xings_list.append(well)
            multiple_xings_tf = True

    # If there are no instances, a placeholder of 0 is supplied
    if multiple_xings_tf == False:
        multiple_crossings = 0
    # Otherwise, unique wells are documented for the user to look into
    else: 
        multiple_crossings = list(set(multiple_xings_list))

    return od_list_intp1d_df, multiple_crossings




def summary_tidy(input_df):

    '''
    Overview / Goal:
    The goal of this function is to clean up the merged output in 
    a standardized way. From my experience of running the code with 
    several modifications requested, it led to very messy outputs 
    that are hard to work with. I thought about merging on several 
    of the keys, but there could be columns with equivalent values, 
    messing up my inner join. One way to clean up the redundant 
    names would be to do so after the merge (I could do so before, 
    but I struggled with getting a reliable result). Therefore, by 
    cleaning it up before it gets printed (getting rid of redundant 
    columns), the output is easier to work with and nicer to look at.

    Argument:
    * input_df: This is the result of merging newly-calculated values 
    to the soon-to-be *_summary.txt output.
    '''

    modified = input_df
    
    # Clearing Previously Stored Data
    col_list = []
    col_counter = 0
    list_to_move = []

    
    for col in modified:
        if col[:2] == "dx":
            pass
        elif col[-2:] == "_x":
            pass
        elif col[-2:] == "_y":
            pass
        elif col[:4] == "Time":
            pass
        else:
            col_list.append(col)
            


    for col in modified[col_list]:
        if col[:15] == "Interpolated_OD":
            list_to_move.append(col)
        col_counter += 1

    list_to_move.append("Sample_ID")
    
    
    
    modified_without_to_move = modified[col_list]

    for i in range(len(list_to_move)-1):
        modified_without_to_move = modified_without_to_move.loc[:, modified_without_to_move.columns != list_to_move[i]]

    modified_with_to_move = modified[list_to_move]
    
    cleaned_df = pd.merge(modified_without_to_move, modified_with_to_move, on = "Sample_ID")
    cleaned_df = cleaned_df.drop_duplicates("Sample_ID").reset_index().drop("index", axis = 1)
    return cleaned_df


    


def summary_tidy_single(input_df):

    '''
    Overview / Goal:

    Argument:
    * input_df: This is the data that are about to be saved.
    I thought that it would be good idea to preclean to avoid
    having to clean the dataframe every time in a Python notebook.
    The removed columns are identical to their not _x/_y forms.
    '''

    # Clearing information from previous run
    list_interpolations = []
    list_interpolations_not = []
    df_to_tidy = input_df

    # This looks for all the interpolated columns and appends
    # them to a list to clean.
    for col in df_to_tidy:
        if col[:13] == "Interpolated_":
            list_interpolations.append(col)
        elif col == "Sample_ID":
            list_interpolations.append(col)
        else:
            pass

    # This finds all the columns that are okay to leave as is
    for col in df_to_tidy:
        if col[:13] == "Interpolated_":
            pass
        else: 
            list_interpolations_not.append(col)

    # Making a DataFrame of only the interpolated values
    tidy_interpolations = df_to_tidy[list_interpolations]
    # Making a DataFrame of other values 
    tidy_interpolations_not = df_to_tidy[list_interpolations_not]
    # Merging the cleaned halves into one whole
    result = pd.merge(tidy_interpolations_not, tidy_interpolations, on = ["Sample_ID"])
    result = result.drop_duplicates("Sample_ID").reset_index().drop("index", axis = 1)

    return result