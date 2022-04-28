#!/usr/bin/env python

'''
AMiGA library of functions for analyzing growth curves.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (9 functions)

# basicSummaryOnly
# runGrowthFitting
# runCombinedGrowthFitting
# handleMissingData
# prepDataForFitting
# savePlots
# prepGpData
# savePlateData
# mergeSummaryData

import pandas as pd
import numpy as np
import tempfile
import operator
import sys
import os

from libs.model import GrowthModel 
from libs.growth import GrowthPlate
from libs.org import assembleFullName, assemblePath
from libs.utils import concatFileDfs, resetNameIndex, subsetDf
from libs.utils import getValue, getTimeUnits, selectFileName, uniqueRandomString
from libs.comm import *
from libs.params import *
from libs.interface import checkParameterCommand


########## Modifications ##########
from libs.config import *  
# In the debugging process, the functions were not updating.
# Using the reload() function reloads each import.
# This solved some significant problems.
import libs.eh_expansion_v3 as eh_imports 
import importlib
importlib.reload(eh_imports)
###################################


def basicSummaryOnly(data,mapping,directory,args,verbose=False):
    '''
    If user only requested plotting, then for  each data file, perform a basic algebraic summary
        and plot data. Once completed, exit system. Otherwise, return None.
 
    Args:
        data (dictionary): keys are plate IDs and values are pandas.DataFrames with size t x (n+1)
            where t is the number of time-points and n is number of wells (i.e. samples),
            the additional 1 is due to the explicit 'Time' column, index is uninformative.
        mapping (dictionary): keys are plate IDs and values are pandas.DataFrames with size n x (p)
            where is the number of wells (or samples) in plate, and p are the number of variables or
            parameters described in dataframe.
        directory (dictionary): keys are folder names, values are their paths
        args (argparse.Namespace class): attribute names are arguments and their value are user/default choices
        verbose (boolean)

    Returns:
        None: if only_plot_plate argument is False. 
    '''

    list_keys = []

    for pid,data_df in data.items():

        # define paths where summary and plot will be saved
        key_file_path = assemblePath(directory['summary'],pid,'.txt')
        key_fig_path = assemblePath(directory['figures'],pid,'.pdf')

        # grab plate-specific samples
        #   index should be well IDs but a      column Well should also exist
        #   in main.py, annotateMappings() is called which ensures the above is the case
        mapping_df = mapping[pid]
        mapping_df = resetNameIndex(mapping_df,'Well',False)

        # grab plate-specific data
        wells = list(mapping_df.Well.values)
        data_df = data_df.loc[:,['Time']+wells]

        # update plate-specific data with unique Sample Identifiers 
        sample_ids = list(mapping_df.index.values)
        data_df.columns = ['Time'] + sample_ids

        # create GrowthPlate object, perform basic summary
        plate = GrowthPlate(data=data_df,key=mapping_df)
        plate.convertTimeUnits(input=getTimeUnits('input'),output=getTimeUnits('output'))
        plate.computeBasicSummary()
        plate.computeFoldChange(subtract_baseline=True)
        plate.subtractControl(to_do=args.subtract_blanks,drop=False,blank=True)
        plate.subtractControl(to_do=args.subtract_control,drop=False,blank=False)

        # plot and save as PDF, also save key as TXT
        if not args.dont_plot: plate.plot(key_fig_path)

        if args.merge_summary: list_keys.append(plate.key) 
        else: plate.key.to_csv(key_file_path,sep='\t',header=True,index=False)

        smartPrint(pid,verbose=verbose)

    if args.merge_summary:
        filename = selectFileName(args.output)
        summary_path = assembleFullName(directory['summary'],'summary',filename,'_basic','.txt')
        summary_df = pd.concat(list_keys,sort=False)
        summary_df.to_csv(summary_path,sep='\t',header=True,index=False)

    smartPrint('\nSee {} for summary text file(s).'.format(directory['summary']),verbose)
    smartPrint('See {} for figure PDF(s).\n'.format(directory['figures']),verbose)

    msg = 'AMiGA completed your request and '
    msg += 'wishes you good luck with the analysis!'
    print(tidyMessage(msg))

    sys.exit()


def runGrowthFitting(data,mapping,directory,args,verbose=False):
    '''
    Uses Gaussian Processes to fit growth curves and infer paramters of growth kinetics.  

    Args:
        data (pandas.DataFrame): number of time points (t) x number of variables plus-one (p+1)
            plus-one because Time is not an index but rather a column.
        mapping (pandas.DataFrame): number of wells/samples (n) x number of variables (p)
        directory (dictionary): keys are folder names, values are their paths
        args (argparse.Namespace class): attribute names are arguments and their value are user/default choices
        verbose (boolean)

    Action:
        saves summary text file(s) in summary folder in the parent directory.
        saves figures (PDFs) in figures folder in the parent directory.
        saves data text file(s) in derived folder in the parent directory.
    '''

    if args.pool_by:
        runCombinedGrowthFitting(data,mapping,directory,args,verbose=verbose)
        return None

    # only store data if user requested its writing or requested plotting
    if args.save_gp_data or args.plot or args.plot_derivative: store = True
    else: store = False

    #if args.output: args.merge_summary = True

    # if user requested merging of summary/data, store each plate's data/summary in temp directory first
    tmpdir = tempfile.mkdtemp()
    saved_umask = os.umask(0o77)  ## files can only be read/written by creator for security
    print('Temporary directory is {}\n'.format(tmpdir))

    # pre-process data
    plate = prepDataForFitting(data,mapping,subtract_baseline=True,drop_flagged_wells=False,
        subtract_control=args.subtract_control,subtract_blanks=args.subtract_blanks,log_transform=args.log_transform)

    dx_ratio_varb = getValue('diauxie_ratio_varb')
    dx_ratio_min = getValue('diauxie_ratio_min')
 
    ls_temp_files = []
    ls_summ_files = []
    ls_diux_files = []

    # for each plate, get samples and save individual text file for plate-specific summaries
    for pid in plate.key.Plate_ID.unique():

        smartPrint('Fitting {}'.format(pid),verbose)

        # grab plate-specific summary
        sub_plate = plate.extractGrowthData(args_dict={'Plate_ID':pid})

        # the primary motivation of this function: run gp model 
        ########## Modifications ##########
        store = True
        ###################################
        sub_plate.model(nthin=args.time_step_size,store=store,verbose=verbose)

        sub_plate = cleanUnlogged(sub_plate,args)

        # save plots, if requested by user
        savePlots(sub_plate,args,directory,pid)
        
        # define file paths where data will be written
        if args.merge_summary or args.output:
            temp_path = assembleFullName(tmpdir,'',pid,'gp_data','.txt')
            summ_path = assembleFullName(tmpdir,'',pid,'summary','.txt') 
            diux_path = assembleFullName(tmpdir,'',pid,'diauxie','.txt')
        else:
            temp_path = assembleFullName(directory['derived'],'',pid,'gp_data','.txt')
            summ_path = assembleFullName(directory['summary'],'',pid,'summary','.txt')
            diux_path = assembleFullName(directory['summary'],'',pid,'diauxie','.txt')
        
        ########## Modifications ##########
        args.save_gp_data = True
        savePlateData(args.save_gp_data,sub_plate,temp_path,summ_path,diux_path)
        overall_path = summ_path
        gp_data = sub_plate.gp_data
        summary_data = sub_plate.key
        opt_in_detection(overall_path)
        duration = eh_imports.duration_extractor(overall_path)


        # Defining a class with attributes- helping me with working with modifications requested.
        class arg_extractor(object): 
            arbitrary_x = "No Value Selected"
            arbitrary_y = "No Value Selected"
            interp1d_kind = "No Value Selected"
            list_times_to_interp1d = "No Value Selected"
            list_ods_to_interp1d = "No Value Selected"

            # __init__ will help define the attributes
            def __init__(self):
                function_input = eh_imports.opt_in_detection(overall_path)[1]

                # Resetting Previous work
                list_counter = 0
                list_times_to_interp1d = []
                list_ods_to_interp1d = []

                # Handling interp1d type
                interp1d_kind = function_input["interp1d_kind"][0]


                # Only dealing with the List Inputs
                for args_list in function_input["list"]:
                    # If the Excel File is left be, the NaNs are read in as floats and the actual numbers are read in as strings
                    if type(args_list).__name__ == "str":
                        # KEY ASSUMPTION- can be modified for later
                        # Each of the values in excel must be split by a comma without a space
                        split = args_list.split(",")
                        # The first list has to do with the times
                        if list_counter == 0:
                            list_times_to_interp1d_float = split
                            # Manually converting the strings to floats
                            for time in list_times_to_interp1d_float:
                                time = list_times_to_interp1d.append(float(time))
                        # The second list has to do with the od values
                        elif list_counter == 1:
                            list_ods_to_interp1d_float = split 
                            # Manually converting the strings to floats
                            for od in list_ods_to_interp1d_float:
                                od = list_ods_to_interp1d.append(float(od))
                        list_counter += 1

                # Handling x and y parameters for single inputs
                arbitrary_y = function_input["arbitrary_y"][2]
                arbitrary_x = function_input["arbitrary_x"][0]
                arbitrary_x = eh_imports.redundancy_checker(arbitrary_x, arbitrary_y, list_times_to_interp1d, list_ods_to_interp1d, duration)[0]
                list_times_to_interp1d = eh_imports.redundancy_checker(arbitrary_x, arbitrary_y, list_times_to_interp1d, list_ods_to_interp1d, duration)[1]
                list_ods_to_interp1d = eh_imports.redundancy_checker(arbitrary_x, arbitrary_y, list_times_to_interp1d, list_ods_to_interp1d, duration)[2]


                ## If I didn't have all the if/else statements, the attributes would return as "nan".
                ## This was difficult to work with, but using NVS instead enabled me to get a clear
                ## prinout of attributes used in the command line to help the user orient themselves.

                # Handling Inputs- making No Value Selected (NVS) Clear
                if np.isnan(arbitrary_x) == True:
                    self.arb_x = "NVS"
                else:
                    self.arb_x = arbitrary_x
                if np.isnan(arbitrary_y) == True:
                    self.arb_y = "NVS"
                else:
                    self.arb_y = arbitrary_y
                try:
                    if np.isnan(interp1d_kind) == True:
                        self.kind = "NVS"
                    else:
                        self.kind = interp1d_kind
                except:
                    self.kind = interp1d_kind
                if np.isnan(list_times_to_interp1d).any() == True:
                    self.list_times == "NVS"
                else:
                    self.list_times = list_times_to_interp1d
                if np.isnan(list_ods_to_interp1d).any() == True:
                    self.list_ods = "NVS"
                else:
                    self.list_ods = list_ods_to_interp1d
        # This indicates that user modifications were requested
        user_mods = True

        interval_extracted = eh_imports.interval_extractor(overall_path)


        ## Initiation of Message
        message = "Additional Arguments"
        print("#" * len(message))
        print(message)
        print("Duration: " + str(duration) + " Hours")
        print("Interval Should be: " + str(interval_extracted) + " Seconds")

        ## Arbitrary X
        try:
            # This relies on the if/else statements made in the __init__ portion of the argextractor class.
            # Values that don't exist there are marked with an "NVS tag"
            if arg_extractor().arb_x != "NVS":
                print("Arbitrary Time:", arg_extractor().arb_x)
            else: 
                print("Arbitrary Time:", "No Value Selected")
        except:
            print("Arbitrary Time:", "None Supplied")

        ## Arbitrary Y
        try:
            if arg_extractor().arb_y != "NVS":
                print("Arbitrary OD Value:", arg_extractor().arb_y)
            else: 
                print("Arbitrary OD Value:", "No Value Selected")
        except:
            print("Arbitrary OD Value:", "None Supplied")

        ## Interpolation Kind
        try:
            if arg_extractor().kind != "NVS":
                print("Type of Interpolation:", arg_extractor().kind)
            else: 
                print("Type of Interpolation:", "No Value Selected")
        except:
            print("Type of Interpolation:", "None Supplied")

        ## List of Times
        try:
            if arg_extractor().list_times != "NVS":
                print("List of Arbitrary Times:", arg_extractor().list_times)
            else: 
                print("List of Arbitrary Times:", "No Values Selected")
        except:
            print("List of Arbitrary Times:", "None Supplied")

        ## List of OD Values
        try:
            if arg_extractor().list_ods != "NVS":
                print("List of Arbitrary OD Values:", arg_extractor().list_ods)
            else: 
                print("List of Arbitrary OD Values:", "No Values Selected")
        except:
            print("List of Arbitrary OD Values:", "None Supplied")

        ## Summary Message
        print("#" * len(message))
        print("\n\n")
        dfs_to_merge = []

        ## Actually performing the additional calculations and merging them to the soon-to-be output
        try: 
            # If nothing exists here, it will produce an error, moving to the except portion
            dfs_to_merge.append(eh_imports.arbitrary_od_finder_time(gp_data, arg_extractor().kind, arg_extractor().arb_x, duration)[0])
            print("Additional Modifications Were Requested for a Time.")
        except:
            print("No Additional Modifications Were Requested for a Time.")
        try: 
            dfs_to_merge.append(eh_imports.arbitrary_od_finder_time_list(gp_data, summary_data, arg_extractor().kind, arg_extractor().list_times, duration))
            print("Additional Modifications Were Requested for a List of Times.")
        except:
            print("No Additional Modifications Were Requested for a List of Times.")
        try: 
            dfs_to_merge.append(eh_imports.arbitrary_od_finder_od(gp_data, arg_extractor().kind, arg_extractor().arb_y)[0])
            print("Additional Modifications Were Requested for an OD Value.")
            multiple_xs = eh_imports.arbitrary_od_finder_od(gp_data, arg_extractor().kind, arg_extractor().arb_y)[2]
            if multiple_xs != 0:
                print("These wells had interpolated OD Values that crossed several times: " + str(multiple_xs))
        except:
            print("No Additional Modifications Were Requested for an OD Value.")
        try: 
            dfs_to_merge.append(eh_imports.arbitrary_od_finder_od_list(gp_data, summary_data, arg_extractor().kind, arg_extractor().list_ods)[0])
            print("Additional Modifications Were Requested for a List of OD Values.")
            multiple_xs = eh_imports.arbitrary_od_finder_od_list(gp_data, summary_data, arg_extractor().kind, arg_extractor().list_ods)[1]
            if multiple_xs != 0:
                print("These wells had interpolated OD Values that crossed several times: " + str(multiple_xs))
        except:
            print("No Additional Modifications Were Requested for a List of OD Values.")


        # Need to merge data
        counter_df = 0
        if len(dfs_to_merge) == 1:
            df_to_merge = dfs_to_merge[0]
            keys_to_join = ["Sample_ID", "Well", "Plate_ID", "Group", "Control", 
            "Flag", "Subset", "OD_Baseline", "OD_Min", "OD_Max", "OD_Emp_AUC", 
            "Fold_Change", "auc_lin", "auc_log", "k_lin", "k_log", "death_lin", 
            "death_log", "gr", "dr", "td", "lagC", "lagP", "t_k", "t_gr", "t_dr", 
            "diauxie", "MSE", "auc_lin", "auc_log", "k_log", "k_lin", "lagC", 
            "lagP", "K_Error > 20%"]
            new_df = pd.merge(df_to_merge, summary_data, on = (keys_to_join), how = "inner")
            updated_df = new_df
            updated_df = eh_imports.summary_tidy_single(updated_df)
        elif len(dfs_to_merge) > 1:
            for df_to_merge in dfs_to_merge:
                if counter_df == 0:
                    new_df = pd.merge(df_to_merge, summary_data, on = ("Sample_ID"), how = "inner")
                    updated_df = new_df
                else:
                    updated_df = pd.merge(df_to_merge, updated_df, on = ("Sample_ID"), how = "inner")
                counter_df += 1
        else:
            pass

        # Tidying of data- preparing to save results
        updated_df = eh_imports.summary_tidy(updated_df)

        

        ## If the user wanted additional modifications, then the updated_df is created.
        ## If they want to use default metrics, then the updated_df does not exist and the variable keeps its data
        try:
            summ_dir, summ_file = os.path.split(overall_path)
            complete_path = os.path.join(summ_dir, "customized_amiga_output_summary.xlsx")
            with pd.ExcelWriter(complete_path,
                mode='w', engine="openpyxl") as writer:
                updated_df.to_excel(writer, sheet_name = "Data", index = False)
            print("Saved Data with Modifications.")
        except:
            # extra emphasis that nothing is changing here
            pass

        # No need to overwrite the gp_data again
        args.save_gp_data = False
        ###################################




        # save data, if requested by user
        savePlateData(args.save_gp_data,sub_plate,temp_path,summ_path,diux_path)

        # track all potentially created files
        ls_temp_files.append(temp_path)
        ls_summ_files.append(summ_path)
        ls_diux_files.append(diux_path)

    # if user did not pass file name for output, use time stamp, see selectFileName()
    filename = selectFileName(args.output)

    # if user requested merging, merge all files in temporary directory
    mergeSummaryData(args,directory,ls_temp_files,ls_summ_files,ls_diux_files,filename)

    # remove temporary directory
    os.umask(saved_umask)
    os.rmdir(tmpdir)

    return None

def cleanUnlogged(plate,args):
    
    if args.do_not_log_transform:

        to_drop = ['auc_lin','k_lin','death_lin','td','dx_auc_lin','dx_k_lin','dx_death_lin','dx_td']
        to_drop = list(set(plate.key.columns).intersection(set(to_drop)))
        plate.key = plate.key.drop(to_drop,axis=1)
        plate.key = plate.key.rename(columns={'auc_log':'auc_lin','k_log':'k_lin','death_log':'death_lin',
            'dx_auc_log':'dx_auc_lin','dx_k_log':'dx_k_lin','dx_death_log':'dx_death_lin'})

    return plate


def runCombinedGrowthFitting(data,mapping,directory,args,verbose=False):
    '''
    Uses Gaussian Processes to fit growth curves and infer paramters of growth kinetics.
        While runGrowthFitting() analyzes data one plate at a time, runCombinedGrowthFitting()
        can pool experimental replicates across different plates. The downside is that data
        summary must be merged and no 96-well plate grid figure can be produced.  

    Args:
        data (pandas.DataFrame): number of time points (t) x number of variables plus-one (p+1)
            plus-one because Time is not an index but rather a column.
        mapping (pandas.DataFrame): number of wells/samples (n) x number of variables (p)
        directory (dictionary): keys are folder names, values are their paths
        args (argparse.Namespace class): attribute names are arguments and their value are user/default choices
        verbose (boolean)

    Action:
        saves summary text file(s) in summary folder in the parent directory.
        saves figures (PDFs) in figures folder in the parent directory.
        saves data text file(s) in derived folder in the parent directory.
    '''   

    # if user did not pass file name for output, use time stamp, see selectFileName()
    filename = selectFileName(args.output)

    # pre-process data
    plate = prepDataForFitting(data,mapping,subtract_baseline=False,drop_flagged_wells=True,
        subtract_control=args.subtract_control,subtract_blanks=args.subtract_blanks,log_transform=args.log_transform)

    # which meta-data variables do you use to group replicates?
    combine_keys = args.pool_by.split(',')
    missing_keys = [ii for ii in combine_keys if ii not in plate.key.columns]

    if missing_keys:
        msg = 'FATAL USER ERROR: The following keys {} are '.format(missing_keys)
        msg += 'missing from mapping files.'
        sys.exit(msg)

    # continue processing data
    plate.subtractBaseline(to_do=True,poly=getValue('PolyFit'),groupby=combine_keys) 
    plate_key = plate.key.copy()
    plate_data = plate.data.copy()
    plate_time = plate.time.copy()
    plate_cond = plate_key.loc[:,combine_keys+['Group','Control']].drop_duplicates(combine_keys).reset_index(drop=True)

    smartPrint('AMiGA detected {} unique conditions.\n'.format(plate_cond.shape[0]),verbose)

    data_ls, diauxie_dict = [], {}

    # get user-defined values from config.py
    dx_ratio_varb = getValue('diauxie_ratio_varb')
    dx_ratio_min = getValue('diauxie_ratio_min')
    posterior_n = getValue('n_posterior_samples')

    posterior = args.sample_posterior
    fix_noise = args.fix_noise
    nthin = args.time_step_size

    # initialize empty dataframes for storing growth parameters
    params_latent = initParamDf(plate_cond.index,complexity=0)
    params_sample = initParamDf(plate_cond.index,complexity=1)
    mse_df = pd.DataFrame(index=plate_cond.index,columns=['MSE'])

    # for each unique condition based on user request
    for idx,condition in plate_cond.iterrows():

        # get list of sample IDs
        cond_dict = condition.drop(['Group','Control'])
        cond_dict = cond_dict.to_dict() # e.g. {'Substate':['D-Trehalose'],'PM':[1]} 
        cond_idx = subsetDf(plate_key,cond_dict).index.values  # list of index values for N samples
        smartPrint('Fitting\n{}'.format(tidyDictPrint(cond_dict)),verbose)

        # get data and format for GP instance
        cond_data = plate_data.loc[:,list(cond_idx)] # T x N
        cond_data = plate_time.join(cond_data) # T x N+1

        # warn user if there is missing data
        cond_data = handleMissingData(args,cond_data)

        cond_data = cond_data.melt(id_vars='Time',var_name='Sample_ID',value_name='OD')
        cond_data = cond_data.drop(['Sample_ID'],axis=1) # T*R x 2 (where R is number of replicates)
        cond_data = cond_data.dropna()
        
        gm = GrowthModel(df=cond_data,ARD=True,heteroscedastic=fix_noise,nthin=nthin,logged=plate.mods.logged)#,

        curve = gm.run(name=idx)
        mse_df.loc[idx,'MSE'] = curve.compute_mse(pooled=True)

        # get parameter estimates using latent function
        diauxie_dict[idx] = curve.params.pop('df_dx')
        params_latent.loc[idx,:] = curve.params

        # get parameter estimates using samples fom the posterior distribution
        if posterior: params_sample.loc[idx,:] = curve.sample().posterior

        # passively save data, manipulation occurs below (input OD, GP fit, & GP derivative)
        if args.save_gp_data:
            time = pd.DataFrame(gm.x_new,columns=['Time'])
            mu0,var0 = np.ravel(gm.y0),np.ravel(np.diag(gm.cov0))
            mu1,var1 = np.ravel(gm.y1),np.ravel(np.diag(gm.cov1))

            if fix_noise: sigma_noise = np.ravel(gm.error_new)+gm.noise
            else: sigma_noise = np.ravel([gm.noise]*time.shape[0])

            mu_var = pd.DataFrame([mu0,var0,mu1,var1,sigma_noise],index=['mu','Sigma','mu1','Sigma1','Noise']).T
            gp_data = pd.DataFrame([list(condition.values)]*len(mu0),columns=condition.keys())
            gp_data = gp_data.join(time).join(mu_var)
            data_ls.append(gp_data)

    # summarize diauxie results
    diauxie_df = mergeDiauxieDfs(diauxie_dict)

    if posterior: gp_params = params_sample.join(params_latent['diauxie'])
    else: gp_params = params_latent

    # record results in object's key
    plate_cond = plate_cond.join(gp_params).join(mse_df)
    plate_cond.index.name = 'Sample_ID'
    plate_cond = plate_cond.reset_index(drop=False)
    plate_cond = pd.merge(plate_cond,diauxie_df,on='Sample_ID')

    params = initParamList(0) + initParamList(1)
    params = list(set(params).intersection(set(plate_cond.keys())))

    df_params = plate_cond.drop(initDiauxieList(),axis=1).drop_duplicates()
    df_diauxie = plate_cond[plate_cond.diauxie==1]
    df_diauxie = df_diauxie.drop(params,axis=1)
    df_diauxie = minimizeDiauxieReport(df_diauxie)

    # because pooling, drop linear AUC, K, and Death 
    to_remove = ['death_lin','k_lin','auc_lin']
    if args.do_not_log_transform: to_remove += ['td']
    
    to_remove = np.ravel([[jj for jj in df_params.keys() if ii in jj] for ii in to_remove])
    df_params.drop(to_remove,axis=1,inplace=True)

    to_remove = np.ravel([[jj for jj in df_diauxie.keys() if ii in jj] for ii in to_remove])
    df_diauxie.drop(to_remove,axis=1,inplace=True)

    if args.do_not_log_transform:
        columns = {'auc_log':'auc_lin','k_log':'k_lin','death_log':'death_lin',
                   'dx_auc_log':'dx_auc_lin','dx_k_log':'dx_k_lin','dx_death_log':'dx_death_lin'}
        df_params.rename(columns=columns,inplace=True)
        df_diauxie.rename(columns=columns,inplace=True)

    summ_path = assembleFullName(directory['summary'],'',filename,'summary','.txt')
    diux_path = assembleFullName(directory['summary'],'',filename,'diauxie','.txt')

    # normalize parameters, if requested
    df_params = minimizeParameterReport(df_params)

    # save results
    df_params.to_csv(summ_path,sep='\t',header=True,index=False)
    if df_diauxie.shape[0]>0:
        df_diauxie.to_csv(diux_path,sep='\t',header=True,index=False)

    # save latent functions
    if args.save_gp_data:
        file_path = assembleFullName(directory['derived'],'',filename,'gp_data','.txt')
        gp_data = pd.concat(data_ls,sort=False).reset_index(drop=True)
        gp_data.to_csv(file_path,sep='\t',header=True,index=True)

    return None


def handleMissingData(args,df):
    '''
    Warns use about missing data and drops it if requested

    Args:
        args (argparse.Namespace class): attribute names are arguments and their value are user/default choices
        df (pandas.DataFrame)

    Returns 
        df (pandas.DataFrame)
    '''

    missing = np.unique(np.where(df.iloc[:,1:].isna())[0])
    missing = df.iloc[missing,].Time.values

    if len(missing) > 0:
        
        msg = 'WARNING: AMiGA detected missing data. Values are missing for some '
        msg += 'replicates at the following time-points:\n'
        msg += ', '.join([str(ii) for ii in missing])
        msg += '\n'
        smartPrint(msg,args.verbose)

    if ~args.keep_missing_time_points: df = df[~df.isna().any(1)]

    return df


def prepDataForFitting(data,mapping,subtract_baseline=True,subtract_control=False,subtract_blanks=False,log_transform=False,drop_flagged_wells=False):
    '''
    Packages data set into a grwoth.GrowthPlate() object and transforms data in preparation for GP fitting.

    Args:
        data (pandas.DataFrame): number of time points (t) x number of variables plus-one (p+1)
            plus-one because Time is not an index but rather a column.
        mapping (pandas.DataFrame): number of wells/samples (n) x number of variables (p)
       
    Returns:
        plate (growth.GrwothPlate() object) 
    '''

    # merge data-sets for easier analysis and perform basic summaries and manipulations
    plate = GrowthPlate(data=data,key=mapping)

    plate.convertTimeUnits(input=getTimeUnits('input'),output=getTimeUnits('output'))
    plate.computeBasicSummary()
    plate.computeFoldChange(subtract_baseline=subtract_baseline)
    plate.subtractControl(to_do=subtract_blanks,drop=getValue('drop_blank_wells'),blank=True)
    plate.subtractControl(to_do=subtract_control,drop=getValue('drop_control_wells'),blank=False)
    plate.raiseData()  # replace non-positive values, necessary prior to log-transformation
    plate.logData(to_do=log_transform)  # natural-log transform
    plate.subtractBaseline(subtract_baseline,poly=False)  # subtract first T0 (or rather divide by first T0)
    plate.dropFlaggedWells(to_do=drop_flagged_wells)

    return plate


def savePlots(plate,args,directory,filename):
    '''
    Saves the GP model fits of grwoth.GrowthPlate() object as a plot. 

    Args:
        plate (growth.GrwothPlate() object)
        args (argparse.Namespace class): attribute names are arguments and their value are user/default choices
        directory (dictionary): keys are folder names, values are their paths
        filename (str): file name

    Returns:
        None
    '''

    if args.plot:  # plot OD and its GP estimate

        fig_path = assembleFullName(directory['figures'],'',filename,'fit','.pdf')
        plate.plot(fig_path,plot_fit=True,plot_derivative=False)

    if args.plot_derivative:  # plot GP estimate of dOD/dt (i.e. derivative)

        fig_path = assembleFullName(directory['figures'],'',filename,'derivative','.pdf')
        plate.plot(fig_path,plot_fit=False,plot_derivative=True)


def prepGpData(plate):
    '''
    Packages different variants of grwoth curves for a specific plate into a single pandas.DataFrame.

    Args:
        plate (growth.GrwothPlate() object)

    Returns: 
        pandas.DataFrame (10 columns). See gp_data attribute in GrwothPlate().__init__() for more info.
    '''

    col_order = ['Plate_ID','Sample_ID','Time','OD_Data','OD_Fit']
    col_order += ['GP_Input','GP_Output','OD_Growth_Data','OD_Growth_Fit','GP_Derivative']

    pids = plate.key.drop_duplicates(subset='Sample_ID').set_index('Sample_ID')
    pids = pids.loc[:,['Plate_ID']]

    data = plate.time.join(plate.input_data)
    data = data.melt(id_vars='Time',var_name='Sample_ID',value_name='OD_Data')
    data = data.join(pids,on='Sample_ID')

    data = pd.merge(data,plate.gp_data,on=['Sample_ID','Time'])
    data = data.sort_values(['Sample_ID','Time'],ascending=[True,True])
    data = data.loc[:,col_order]

    return data


def savePlateData(store_data,plate,data_path,summ_path,diux_path):
    '''
    Saves the 'gp_data' and 'key' (attributes of 'plate' object) as tab-delimited files. 

    Args:
        store_data (boolean): whether to run part of code or not (save data from gp-model)
        plate (growth.GrwothPlate() object)
        data_path (str): file path for storing data file
        summ_path (str): file path for storing summary file
    '''

    #params = getValue('params_report')
    params = initParamList(0) + initParamList(1) + initParamList(2) 
    params = list(set(params).intersection(set(plate.key.keys())))
    diauxie = list(set(plate.key.keys()).intersection(set(initDiauxieList())))

    df_params = plate.key.drop(diauxie,axis=1).drop_duplicates().reset_index(drop=True)
    df_params = minimizeParameterReport(df_params)
    df_params = df_params.drop_duplicates()

    df_diauxie = plate.key[plate.key.diauxie==1]
    df_diauxie = df_diauxie.drop(params,axis=1)
    df_diauxie = df_diauxie.loc[:,~df_diauxie.columns.str.startswith('norm_')]
    df_diauxie = df_diauxie.reset_index(drop=True)
    df_diauxie = minimizeDiauxieReport(df_diauxie)


    df_params.drop(['Subset'],axis=1,inplace=True)
    df_params.to_csv(summ_path,sep='\t',header=True,index=False)
    if df_diauxie.shape[0]>0:
        df_diauxie.drop(['Subset'],axis=1,inplace=True)
        df_diauxie.to_csv(diux_path,sep='\t',header=True,index=False)

    if not store_data:
        return None

    prepGpData(plate).to_csv(data_path,sep='\t',header=True,index=True)


def mergeSummaryData(args,directory,ls_temp_files,ls_summ_files,ls_diux_files,filename):
    '''
    Reads files passed via list and concatenates them into a single pandas.DataFrame. This 
        is executed for summary files and gp_data files separately. 

    Args:
        args (argparse.Namespace class): attribute names are arguments and their value are user/default choices
        directory (dictionary): keys are folder names, values are their paths
        ls_temp_files (list): where each item is a file path (str)
        ls_temp_files (list): where each item is a file path (str)
        filename (str): base file name  
    '''

    if args.save_gp_data and (args.merge_summary or args.output):

        gp_data_df = concatFileDfs(ls_temp_files)

        gp_data_path = assembleFullName(directory['derived'],'',filename,'gp_data','.txt')

        gp_data_df.to_csv(gp_data_path,sep='\t',header=True,index=True)

    if args.merge_summary or args.output:

        summ_df = concatFileDfs(ls_summ_files)
        diux_df = concatFileDfs(ls_diux_files)

        summ_path = assembleFullName(directory['summary'],'',filename,'summary','.txt')
        diux_path = assembleFullName(directory['summary'],'',filename,'diauxie','.txt')

        summ_df.to_csv(summ_path,sep='\t',header=True,index=True)
        if diux_df.shape[0]>0: diux_df.to_csv(diux_path,sep='\t',header=True,index=True)

        # clean-up
        for f in ls_temp_files + ls_summ_files + ls_diux_files:
            if os.path.isfile(f):
                os.remove(f)

