import pandas as pd
import seaborn as sns
import seaborn
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.dates as mdates
from matplotlib.pylab import datestr2num
from datetime import datetime

STGCN_MU_6_speed = pd.read_csv('results/RichmondResultsALL/richmond_GCNRWZ_6_label.csv')
STGCN_MU_6_pred = pd.read_csv('results/RichmondResultsALL/richmond_GCNRWZ_6_pred.csv')
STGCN_MU_6_pred_NO = pd.read_csv('results/RichmondResultsALL/DGCRN_6_pred.csv')

STGCN_MU_3 = pd.read_csv('results/RichmondResultsALL/trainingInfo_GCNRWZ_3.csv')
STGCN_MU_3NO = pd.read_csv('results/RichmondResultsALL/trainingInfo_GCNRWZ_NO_3.csv')
STGCN_MU_6 = pd.read_csv('results/RichmondResultsALL/trainingInfo_GCNRWZ_6.csv')
STGCN_MU_6NO = pd.read_csv('results/RichmondResultsALL/trainingInfo_GCNRWZ_NO_6.csv')
STGCN_MU_12 = pd.read_csv('results/RichmondResultsALL/trainingInfo_GCNRWZ_12.csv')
STGCN_MU_12NO = pd.read_csv('results/RichmondResultsALL/trainingInfo_GCNRWZ_NO_12.csv')

# heatmap = pd.read_csv("results/RichmondResultsALL/DGCRN_heatmap.csv").T
# a = heatmap.sample(n = 8, random_state = 1234)
# f, ax = plt.subplots(figsize=(8, 8))
# sns.heatmap(a, cmap="YlGnBu", vmin=1, vmax=6, xticklabels = np.array([15, 30, 45, 60, 75, 90]), annot=True)
# plt.xticks(fontsize = 13)
# plt.yticks(fontsize = 11)
# font1 = {'size': 15}
# plt.savefig('picture/heatmap_dgcrn.jpg',bbox_inches='tight')
# plt.show()
#
# heatmap = pd.read_csv("results/RichmondResultsALL/heatmap.csv").T
# a = heatmap.sample(n = 8, random_state = 1234)
# f, ax = plt.subplots(figsize=(8, 8))
# sns.heatmap(a, cmap="YlGnBu", vmin=1, vmax=6, xticklabels = np.array([15, 30, 45, 60, 75, 90]), annot=True)
# plt.xticks(fontsize = 13)
# plt.yticks(fontsize = 11)
# font1 = {'size': 15}
# plt.savefig('picture/heatmap_gcnrwz.jpg',bbox_inches='tight')
# plt.show()
#
STGCN_MU_6_speed['time'] = pd.to_datetime(STGCN_MU_6_speed['time'])
STGCN_MU_6_pred['time'] = pd.to_datetime(STGCN_MU_6_pred['time'])
STGCN_MU_6_pred_NO['time'] = pd.to_datetime(STGCN_MU_6_pred_NO['time'])
s = STGCN_MU_6_speed.iloc[240:480]
p = STGCN_MU_6_pred.iloc[240:480]
d = STGCN_MU_6_pred_NO[240:480]
x = range(len(s))
x_date = s.time

fig = plt.figure(figsize=(12, 5))
font1 = {'size': 15}

ax = fig.add_subplot(1,1,1)
monthly_locator = mdates.MonthLocator()
half_year_locator = mdates.HourLocator(interval=12)
month_year_formatter = mdates.DateFormatter('%D-%H:%M')
ax.xaxis.set_major_locator(half_year_locator)
ax.xaxis.set_minor_locator(monthly_locator)
ax.xaxis.set_major_formatter(month_year_formatter)

a = '140315'
plt.xlabel("(d) Time; Road segment = " +  a, font1)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 15)
plt.ylabel("Speed (MPH)", font1)

# plt.tick_params(labelsize=12)
plt.plot_date(x_date,s[a],'-',label="Ground Truth", color = 'black')
plt.plot_date(x_date,p[a],'--',label="GCN-RWZ", color = 'red')
plt.plot_date(x_date,d[a],'--',label="DGCRN", color = 'teal')
plt.ylim(30, 65, font1)
plt.legend(prop={'size': 12})

plt.savefig('picture/'+ a + '(wz).jpg',bbox_inches='tight')
plt.show()

# plt.plot( 'epoch', 'training_rmse', label = r'$P_{train} = 3$', data=STGCN_MU_3, color = 'red')
# plt.plot( 'epoch', 'training_rmse', label = r'$P_{train} = 6$', data=STGCN_MU_6, color = 'orange')
# plt.plot( 'epoch', 'training_rmse', label = r'$P_{train} = 12$', data=STGCN_MU_12, color = 'green')
# plt.plot( 'epoch', 'validation_rmse', label = r'$P_{val} = 3$', data=STGCN_MU_3, linestyle='dashed', color = 'red')
# plt.plot( 'epoch', 'validation_rmse', label = r'$P_{val} = 6$', data=STGCN_MU_6, linestyle='dashed', color = 'orange')
# plt.plot( 'epoch', 'validation_rmse', label = r'$P_{val} = 12$', data=STGCN_MU_12, linestyle='dashed', color = 'green')
# plt.xlabel('Training Epoch', fontsize=15)
# plt.ylabel('RMSE (MPH)', fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.ylim(3.5, 9)
# plt.legend(prop={'size': 13})
# plt.savefig('picture/STGCN_WZ.jpg',bbox_inches='tight')
# plt.show()

# fig = plt.figure(figsize=(6, 6))
# plt.plot( 'epoch', 'validation_rmse', label = '$P_{val} = 3$ (Normal)', data=STGCN_MU_3, color = "red")
# plt.plot( 'epoch', 'cvalidation_rmse', label = '$P_{val} = 3$ (WZ)' , data=STGCN_MU_3, linestyle='dashed', color = "red")
# plt.plot( 'epoch', 'validation_rmse', label = '$P_{val} = 6$ (Normal)', data=STGCN_MU_6, color = "green")
# plt.plot( 'epoch', 'cvalidation_rmse', label = '$P_{val} = 6$ (WZ)', data=STGCN_MU_6, linestyle='dashed', color = "green")
# plt.plot( 'epoch', 'validation_rmse', label = '$P_{val} = 12$ (Normal)', data=STGCN_MU_12, color = "blue")
# plt.plot( 'epoch', 'cvalidation_rmse', label = '$P_{val} = 12$ (WZ)', data=STGCN_MU_12, linestyle='dashed', color = "blue")
# plt.ylim(2.5, 4.5)
# plt.xlim(-1, 40)
# plt.xlabel('Epoch', fontsize=15)
# plt.ylabel('Validation_RMSE', fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(prop={'size': 10})
# plt.savefig('picture/validation_rmse.jpg',bbox_inches='tight')
# plt.show()

# plt.plot( 'epoch', 'running_time', label = 'STGCN_MU_6', data=STGCN_MU_6, linestyle='dashed')
# plt.plot( 'epoch', 'running_time', label = 'ASTGCN_6', data=ASTGCN_6, linestyle='dashed')
# plt.xlabel('training epoch')
# plt.ylabel('training time')
#
# plt.legend()
# plt.savefig('picture/STGCN_MU_training_time.jpg')
# plt.show()
# #
# # TGCN = {'index':['TGCN'],'3_rmse': [5.66],'3_mae': [3.42],'3_mape': [0.12],'6_rmse': [6.24],'6_mae':[3.62],'6_mape':[0.1357],'9_rmse': [ 7.82],'9_mae': [5.33],'9_mape': [0.16]}
# # ASTGCN = {'index':['ASTGCN'],'3_rmse': [4.18 ],'3_mae': [2.7490],'3_mape': [0.0865],'6_rmse': [5.08],'6_mae':[3.1641],'6_mape':[0.1037],'9_rmse': [6.00],'9_mae':[3.5749],'9_mape': [0.1121]}
# # STGCN = {'index':['STGCN'],'3_rmse':[5.30],'3_mae': [3.2957],'3_mape': [0.1001],'6_rmse': [ 6.19],'6_mae':[3.4207],'6_mape':[0.1107],'9_rmse': [7.28],'9_mae': [4.2532],'9_mape': [0.1332]}
# # GWNET = {'index':['GWNET'],'3_rmse': [4.53],'3_mae': [3.0173],'3_mape':[0.0904],'6_rmse': [5.82],'6_mae':[3.5207],'6_mape': [0.1168],'9_rmse': [6.90],'9_mae': [4.3532],'9_mape': [0.1387]}
# # STGCN_WZ= {'index':['STGCN_WZ'],'3_rmse': [3.95],'3_mae': [2.6023],'3_mape': [0.0820],'6_rmse': [4.79],'6_mae':[3.0844],'6_mape':[0.0954],'9_rmse': [5.86],'9_mae': [3.5138],'9_mape': [0.1065]}
# # STGCN_CON = {'index':['STGCN_WZ(NO)'],'3_rmse': [4.04],'3_mae': [2.6805],'3_mape': [0.0837],'6_rmse': [4.91],'6_mae':[3.0880],'6_mape': [0.1079],'9_rmse': [5.95],'9_mae': [3.5622],'9_mape': [0.1093]}
# # df = pd.DataFrame(data=TGCN)
# # df1 = pd.DataFrame(data=ASTGCN)
# # df2 = pd.DataFrame(data=STGCN)
# # df3 = pd.DataFrame(data=GWNET)
# # df4 = pd.DataFrame(data=STGCN_WZ)
# # df5 = pd.DataFrame(data=STGCN_CON)
# # a = pd.concat([df,df1,df2,df3,df4,df5])
# #
# # x = np.arange(3)  # the label locations
# # width = 0.145 # the width of the bars
# #
# # a1 = [a['3_rmse'].iloc[0],a['6_rmse'].iloc[0],a['9_rmse'].iloc[0]] # TGCN
# # a2 = [a['3_rmse'].iloc[3],a['6_rmse'].iloc[3],a['9_rmse'].iloc[3]] # GWNET
# # a3 = [a['3_rmse'].iloc[2],a['6_rmse'].iloc[2],a['9_rmse'].iloc[2]] # STGCN
# # a4 = [a['3_rmse'].iloc[1], a['6_rmse'].iloc[1],a['9_rmse'].iloc[1]] #ASTGCN
# # a6 = [a['3_rmse'].iloc[5],a['6_rmse'].iloc[5],a['9_rmse'].iloc[5]] # CON
# # a5 = [a['3_rmse'].iloc[4],a['6_rmse'].iloc[4],a['9_rmse'].iloc[4]] # WZ
# #
# # x1 = np.arange(6) + 1
# # y1 = np.array(list(a1))
# # y2 = np.array(list(a2))
# # y3 = np.array(list(a3))
# # y4 = np.array(list(a4))
# # y5 = np.array(list(a5))
# # y6 = np.array(list(a6))
# #
# # fig, ax = plt.subplots(figsize=(6, 6))
# # rects1 = ax.bar(x - 5 * width/2, a1, width, label='T-GCN', color = 'orchid')
# # rects3 = ax.bar(x - 3 * width/2, a3, width, label='STGCN')
# # rects2 = ax.bar(x - 1 * width/2, a2, width, label='GWNET')
# # rects4 = ax.bar(x + 1 * width/2, a4, width, label='ASTGCN')
# # rects5 = ax.bar(x + 3 * width/2, a6, width, label=r'GCN-RWZ$^-$')
# # rects6 = ax.bar(x + 5 * width/2, a5, width, label='GCN-RWZ')
# #
# # # for rect in rects1:
# # #     height = rect.get_height()
# # #     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height), ha="center", va="bottom",fontsize=7.3)
# # # for rect in rects2:
# # #     height = rect.get_height()
# # #     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height), ha="center", va="bottom",fontsize=7.3)
# # # for rect in rects3:
# # #     height = rect.get_height()
# # #     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height), ha="center", va="bottom",fontsize=7.3)
# # # for rect in rects4:
# # #     height = rect.get_height()
# # #     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height), ha="center", va="bottom",fontsize=7.3)
# # # for rect in rects5:
# # #     height = rect.get_height()
# # #     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height), ha="center", va="bottom",fontsize=7.3)
# # # for rect in rects6:
# # #     height = rect.get_height()
# # #     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height), ha="center", va="bottom",fontsize=7.3)
# #
# # plt.xticks(range(0, 3), ['15 minutes', '30 minutes', '60 minutes'], fontsize=15)
# # plt.ylim(3.5, 8.5)
# # plt.ylabel('RMSE (MPH)', fontsize=15)
# # plt.xticks(fontsize=15)
# # plt.yticks(fontsize=15)
# # plt.legend(prop={'size': 13})
# # plt.savefig('picture/RMSE_tyson.jpg',bbox_inches='tight')
# # plt.show()
# # #
# # #

# tyson
# STGCN = {'index':['STGCN'],'3_rmse':[82.24],'6_rmse': [79.94], '9_rmse': [78.03]}
# GWNET = {'index':['GWNET'],'3_rmse':[81.6],'6_rmse': [82.3], '9_rmse': [80.6]}
# ASTGCN = {'index':['ASTGCN'],'3_rmse':[84.11],'6_rmse': [83.09], '9_rmse': [81.4]}
# DGCRN = {'index':['DGCRN'],'3_rmse':[83.43],'6_rmse': [82.5], '9_rmse': [80.8]}
# GCN_RWZ= {'index':['GCN_RWZ'],'3_rmse':[83.52],'6_rmse': [80.49], '9_rmse': [78.55]}
# GCN_RWZ_WZ= {'index':['GCN_RWZ_WZ'],'3_rmse':[85.0],'6_rmse': [82.7], '9_rmse': [81.0]}

# richmond_clean
# STGCN = {'index':['STGCN'],'3_rmse':[81.9],'6_rmse': [80.0], '9_rmse': [78.0]}
# GWNET = {'index':['GWNET'],'3_rmse':[83.9],'6_rmse': [81.5], '9_rmse': [79.5]}
# ASTGCN = {'index':['ASTGCN'],'3_rmse':[84.1],'6_rmse': [82.4], '9_rmse': [82.5]}
# DGCRN = {'index':['DGCRN'],'3_rmse':[85.9],'6_rmse': [84.5], '9_rmse': [83.1]}
# GCN_RWZ= {'index':['GCN_RWZ'],'3_rmse':[87.0],'6_rmse': [85.5], '9_rmse': [83.7]}
# GCN_RWZ_WZ= {'index':['GCN_RWZ_WZ'],'3_rmse':[87.7],'6_rmse': [86.7], '9_rmse': [85.0]}
# richmond_wz
# STGCN = {'index':['STGCN'],'3_rmse':[83.51],'6_rmse': [81.87], '9_rmse': [80.28]}
# GWNET = {'index':['GWNET'],'3_rmse':[85.69],'6_rmse': [83.21], '9_rmse': [81.82]}
# ASTGCN = {'index':['ASTGCN'],'3_rmse':[86.01],'6_rmse': [84.32], '9_rmse': [84.60]}
# DGCRN = {'index':['DGCRN'],'3_rmse':[87.50],'6_rmse': [86.20], '9_rmse': [85.28]}
# GCN_RWZ= {'index':['GCN_RWZ'],'3_rmse':[88.76],'6_rmse': [87.16], '9_rmse': [85.72]}
# GCN_RWZ_WZ= {'index':['GCN_RWZ_WZ'],'3_rmse':[89.45],'6_rmse': [88.39], '9_rmse': [87.33]}

# df1 = pd.DataFrame(data=STGCN)
# df2 = pd.DataFrame(data=GWNET)
# df3 = pd.DataFrame(data=ASTGCN)
# df4 = pd.DataFrame(data=DGCRN)
# df5 = pd.DataFrame(data=GCN_RWZ)
# df6 = pd.DataFrame(data=GCN_RWZ_WZ)
# a = pd.concat([df1, df2, df3, df4, df5, df6])
#
# x = np.arange(3)  # the label locations
# width = 0.15 # the width of the bars
#
# a1 = [a['3_rmse'].iloc[0],a['6_rmse'].iloc[0],a['9_rmse'].iloc[0]]
# a2 = [a['3_rmse'].iloc[1],a['6_rmse'].iloc[1],a['9_rmse'].iloc[1]]
# a3 = [a['3_rmse'].iloc[2],a['6_rmse'].iloc[2],a['9_rmse'].iloc[2]]
# a4 = [a['3_rmse'].iloc[3],a['6_rmse'].iloc[3],a['9_rmse'].iloc[3]]
# #a5 = [a['3_rmse'].iloc[4],a['6_rmse'].iloc[4],a['9_rmse'].iloc[4]]
# a6 = [a['3_rmse'].iloc[5],a['6_rmse'].iloc[5],a['9_rmse'].iloc[5]]
#
# x1 = np.arange(6) + 1
# y1 = np.array(list(a1))
# y2 = np.array(list(a2))
# y3 = np.array(list(a3))
# y4 = np.array(list(a4))
# #y5 = np.array(list(a5))
# y6 = np.array(list(a6))
#
# fig, ax = plt.subplots(figsize=(8, 6))
# rects1 = ax.bar(x - 4 * width/2, a1, width, label='STGCN', color = 'orchid')
# rects2 = ax.bar(x - 2 * width/2, a2, width, label='GWNET')
# rects3 = ax.bar(x - 0 * width/2, a3, width, label='ASTGCN')
# rects4 = ax.bar(x + 2 * width/2, a4, width, label='DGCRN')
# #rects5 = ax.bar(x + 4 * width/2, a5, width, label=r'GCN-RWZ$^-$')
# rects6 = ax.bar(x + 4 * width/2, a6, width, label='GCN-RWZ')
#
# for rect in rects1:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.1f'%height), ha="center", va="bottom",fontsize=8.5)
# for rect in rects2:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.1f'%height), ha="center", va="bottom",fontsize=8.5)
# for rect in rects3:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.1f'%height), ha="center", va="bottom",fontsize=8.5)
# for rect in rects4:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.1f'%height), ha="center", va="bottom",fontsize=8.5)
# #for rect in rects5:
# #    height = rect.get_height()
# #    plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.1f'%height), ha="center", va="bottom",fontsize=8.5)
# for rect in rects6:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.1f'%height), ha="center", va="bottom",fontsize=8.5)
#
# plt.xticks(range(0, 3), ['15 minutes', '30 minutes', '60 minutes'], fontsize=15)
# plt.ylim(65, 100)
# plt.ylabel('Accuracy (%)', fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(prop={'size': 11})
# plt.savefig('picture/Accuracy_tyson_wz.jpg',bbox_inches='tight')
# plt.show()

# TGCN = {'index':['TGCN'],'3_rmse': [3.02],'3_mae': [4.1618],'3_mape': [0.1104],'6_rmse': [4.31],'6_mae':[4.8187],'6_mape':[0.1374],'9_rmse': [5.71],'9_mae': [5.4124],'9_mape': [0.1504]}
# ASTGCN = {'index':['ASTGCN'],'3_rmse': [2.76],'3_mae': [3.0443],'3_mape': [0.0787],'6_rmse': [3.78],'6_mae':[3.5061],'6_mape':[0.1011],'9_rmse': [4.76],'9_mae':[4.4192],'9_mape': [0.1289]}
# STGCN = {'index':['STGCN'],'3_rmse':[2.97],'3_mae': [4.1805],'3_mape': [0.1192],'6_rmse': [4.28],'6_mae':[4.7357],'6_mape':[0.1253],'9_rmse': [5.67],'9_mae': [5.3654],'9_mape': [0.1414]}
# GWNET = {'index':['GWNET'],'3_rmse': [2.75],'3_mae': [3.0042],'3_mape':[0.0977],'6_rmse': [3.67],'6_mae':[3.7182],'6_mape': [0.1235],'9_rmse': [4.74],'9_mae': [4.7148],'9_mape': [0.1400]}
# STGCN_WZ= {'index':['STGCN_WZ(NO)'],'3_rmse': [2.67],'3_mae': [2.9429],'3_mape': [0.0776],'6_rmse': [3.68],'6_mae':[3.2782],'6_mape':[0.0907],'9_rmse': [4.73],'9_mae': [4.3831],'9_mape': [0.1271]}
#
# df = pd.DataFrame(data=TGCN)
# df1 = pd.DataFrame(data=ASTGCN)
# df2 = pd.DataFrame(data=STGCN)
# df3 = pd.DataFrame(data=GWNET)
# df4 = pd.DataFrame(data=STGCN_WZ)
# #df5 = pd.DataFrame(data=STGCN_CON)
# a = pd.concat([df,df1,df2,df3,df4])
#
# x = np.arange(3)  # the label locations
# width = 0.15 # the width of the bars
#
# a1 = [a['3_rmse'].iloc[0],a['6_rmse'].iloc[0],a['9_rmse'].iloc[0]]
# a2 = [a['3_rmse'].iloc[3],a['6_rmse'].iloc[3],a['9_rmse'].iloc[3]]
# a3 = [a['3_rmse'].iloc[2],a['6_rmse'].iloc[2],a['9_rmse'].iloc[2]]
# a4 = [a['3_rmse'].iloc[1],a['6_rmse'].iloc[1],a['9_rmse'].iloc[1]]
# a5 = [a['3_rmse'].iloc[4],a['6_rmse'].iloc[4],a['9_rmse'].iloc[4]]
# #a6 = [a['3_rmse'].iloc[5],a['6_rmse'].iloc[5],a['9_rmse'].iloc[5]]
#
# x1 = np.arange(6) + 1
# y1 = np.array(list(a1))
# y2 = np.array(list(a2))
# y3 = np.array(list(a3))
# y4 = np.array(list(a4))
# y5 = np.array(list(a5))
#
# fig, ax = plt.subplots(figsize=(6, 6))
# rects1 = ax.bar(x - 4 * width/2, a1, width, label='T-GCN', color = 'orchid')
# rects3 = ax.bar(x - 2 * width/2, a3, width, label='STGCN')
# rects2 = ax.bar(x - 0 * width/2, a2, width, label='GWNET')
# rects4 = ax.bar(x + 2 * width/2, a4, width, label='ASTGCN')
# rects5 = ax.bar(x + 4 * width/2, a5, width, label=r'GCN-RWZ$^-$')
# #rects6 = ax.bar(x + 5 * width/2, a6, width, label='STGCN_WZ(NO)')
# # for rect in rects1:
# #     height = rect.get_height()
# #     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height), ha="center", va="bottom",fontsize=8)
# # for rect in rects2:
# #     height = rect.get_height()
# #     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height), ha="center", va="bottom",fontsize=8)
# # for rect in rects3:
# #     height = rect.get_height()
# #     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height), ha="center", va="bottom",fontsize=8)
# # for rect in rects4:
# #     height = rect.get_height()
# #     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height), ha="center", va="bottom",fontsize=8)
# # for rect in rects5:
# #     height = rect.get_height()
# #     plt.text(rect.get_x() + rect.get_width() / 2, height, str('%.2f'%height), ha="center", va="bottom",fontsize=8)
#
# plt.xticks(range(0, 3), ['15 minutes', '30 minutes', '60 minutes'], fontsize=15)
# plt.ylim(1, 6.5)
# plt.ylabel('RMSE (MPH)', fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(prop={'size': 13})
# plt.savefig('picture/RMSE_PEMS.jpg',bbox_inches='tight')
# plt.show()