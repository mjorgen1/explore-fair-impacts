from visualizations import impact_bar_plots
import numpy as np
import pandas as pd





def impact_csvs(data_path= 'data/results/',b_or_w = 'black', folders= ['dt','gnb','lgr','gbt']):

    col_names_eg = []
    col_names_gs = []

    for i,f in enumerate(folders):
        if b_or_w == 'black':
            path = f'{data_path}{f}/{f}_black_results.csv'
        else:
            path = f'{data_path}{f}/{f}_white_results.csv'
        df = pd.read_csv(path,index_col=0)
        df = df.reset_index()

        col_names_eg.append(f'EG+{f.upper()}')
        col_names_gs.append(f'GS+{f.upper()}')

        if i == 0:
            joined_df = df.iloc[:,-1]
        else:
            joined_df = pd.concat([joined_df, df.iloc[:,-1]], axis=1)

    joined_df.set_axis(folders, axis=1)


    # split dataframe after the two reduction algorithms
    df_eg = joined_df.iloc[:6,:]
    df_gs = pd.concat([joined_df.iloc[0:1,:],joined_df.iloc[6:,:]])

    # set new index
    df_eg['Constraint'] = ['Unmitigated', 'DP', 'EO', 'EOO','FPER','ERP']
    df_eg.set_index('Constraint',inplace=True)
    df_gs['Constraint'] = ['Unmitigated', 'DP', 'EO', 'EOO','FPER','ERP']
    df_gs.set_index('Constraint',inplace=True)
    df_eg.columns = col_names_eg
    df_gs.columns = col_names_gs

    df_final = pd.concat([df_eg, df_gs], axis=1)
    print('Group: ',b_or_w,'\n DataFrame: \n',df_final)
    print('A')
    df_final.to_csv(f'{data_path}/{b_or_w}_DI.csv')
    print('B')


def types_ratios_csv(data_path,folders= ['dt','lgr','gbt','gnb']):

    pd.set_option('display.max_columns', None)
    dfs = {} # list for pandas dfs
    for i,f in enumerate(folders):
        path = f'{data_path}{f}/{f}_all_types.csv'
        df = pd.read_csv(path,)
        df = df.reset_index(drop=True)
        #df = df.iloc[:,2:]
        df = df.melt(var_name="ID",value_name="Value")
        df = df.groupby('ID').value_counts(normalize=True)
        df = df.reset_index()
        df = df.rename(columns= {0:'Ratio'})
        df = df.pivot(index='Value', columns='ID')['Ratio']
        print('Classifier: ',f,'\n DataFrame: \n',df)
        df.to_csv(f'{data_path}{f}/{f}_typeRatios.csv')



if __name__ == "__main__":
    impact_bar_plots(data_path = 'data/results/test_1/',b_or_w = 'black')
    impact_bar_plots(data_path = 'data/results/test_1/',b_or_w = 'white')
