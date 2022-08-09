from visualizations import impact_bar_plots
import numpy as np
import pandas as pd

impact_bar_plots(data_path = 'data/results/',b_or_w = 'black')
impact_bar_plots(data_path = 'data/results/',b_or_w = 'white')

def impact_charts(data_path= 'data/results/',b_or_w = 'black', folders= ['dt','lgr','gbt','gnb']):

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




    df_eg = joined_df.iloc[:6,:]
    df_gs = pd.concat([joined_df.iloc[0:1,:],joined_df.iloc[6:,:]])

    df_eg.rename(columns=col_names_eg, index=['Unmitigated', 'DP', 'EO', 'EOO','FPER','ERP'])
    #df_eg.set_names(['Unmitigated', 'DP', 'EO', 'EOO','FPER','ERP'])
    #df_eg.set_axis(col_names_eg, axis=1)
    print(df_eg)
    print(df_gs)
    df_gs.set_names(['Unmitigated', 'DP', 'EO', 'EOO','FPER','ERP'])
    df_gs.set_axis(col_names_gs, axis=1)



    df_final = pd.concat([df_eg, df_gs], axis=1)
    print(df_final)
        #dfs[i] = list(df.iloc[:,-1])

#impact_charts()
