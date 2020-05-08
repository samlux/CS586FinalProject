import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

path_to_raw = 'COVID-19/csse_covid_19_data/csse_covid_19_time_series/'
raw_name = 'time_series_covid19_confirmed_US.csv'

data = pd.read_csv(path_to_raw+raw_name)
nj = []
ny = []
ca = []
tx = []
wa = []
fl = []
ma = []

nj_pop = 8882190
ny_pop = 19453561
ca_pop = 39512223
tx_pop = 28995881
wa_pop = 7614893
fl_pop = 21477737
ma_pop = 6949503

#population density is persons per square mile
nj_dense = nj_pop / 8722.58
ny_dense = ny_pop / 54554.98
ca_dense = ca_pop / 163694.74
tx_dense = tx_pop / 268596.46
wa_dense = wa_pop / 71297.95
fl_dense = fl_pop / 65757.70
ma_dense = ma_pop / 10554.39

nj_area = 8722.58
ny_area = 54554.98
ca_area = 163694.74
tx_area = 268596.46
wa_area = 71297.95
fl_area = 65757.70
ma_area = 10554.39

for index, row in data.iterrows():
    if row['Province_State'] == 'New Jersey':
        nj.append(row)
    elif row['Province_State'] == 'New York':
        ny.append(row)
    elif row['Province_State'] == 'California':
        ca.append(row)
    elif row['Province_State'] == 'Texas':
        tx.append(row)
    elif row['Province_State'] == 'Washington':
        wa.append(row)
    elif row['Province_State'] == 'Florida':
        fl.append(row)
    elif row['Province_State'] == 'Massachusetts':
        ma.append(row)

#print(nj_dense, ny_dense, ca_dense, wa_dense, tx_dense, fl_dense, ma_dense)
days = data.shape[1] - 11
#print('Days:', days)

Days = np.linspace(start = 1, stop = days, num = days, dtype = int)
# NJ
NJ = np.zeros(days)
NJ_cases = []
NJ_POP = []
NJ_AREA = []
NJ_pop= []
NJ_pop_area = []

for i in range(len(nj)):
    nj[i].to_numpy()
for i in range(len(nj)):
    for day in range(days):
        d = day + 11
        NJ[day] += nj[i][d]
for day in range(days):
    if NJ[day] != 0:
        NJ_cases.append(NJ[day])
        NJ_POP.append(nj_pop)
        NJ_AREA.append(nj_area)
        NJ_pop.append(NJ[day] / nj_pop)
        NJ_pop_area.append(NJ[day] / nj_area)

NJ_cases = np.array(NJ_cases)
NJ_POP = np.array(NJ_POP)
NJ_AREA = np.array(NJ_AREA)
NJ_pop = np.array(NJ_pop)
NJ_pop_area = np.array(NJ_pop_area)
# NY
NY = np.zeros(days)
NY_cases = []
NY_POP = []
NY_AREA = []
NY_pop= []
NY_pop_area = []

for i in range(len(ny)):
    ny[i].to_numpy()
for i in range(len(ny)):
    for day in range(days):
        d = day + 11
        NY[day] += ny[i][d]
for day in range(days):
    if NY[day] != 0:
        NY_cases.append(NY[day])
        NY_POP.append(ny_pop)
        NY_AREA.append(ny_area)
        NY_pop.append(NY[day] / ny_pop)
        NY_pop_area.append(NY[day] / ny_area)

NY_cases = np.array(NY_cases)
NY_POP = np.array(NY_POP)
NY_AREA = np.array(NY_AREA)
NY_pop = np.array(NY_pop)
NY_pop_area = np.array(NY_pop_area)
# CA
CA = np.zeros(days)
CA_cases = []
CA_POP = []
CA_AREA = []
CA_pop= []
CA_pop_area = []

for i in range(len(ca)):
    ca[i].to_numpy()
for i in range(len(ca)):
    for day in range(days):
        d = day + 11
        CA[day] += ca[i][d]
for day in range(days):
    if CA[day] != 0:
        CA_cases.append(CA[day])
        CA_POP.append(ca_pop)
        CA_AREA.append(ca_area)
        CA_pop.append(CA[day] / ca_pop)
        CA_pop_area.append(CA[day] / ca_area)

CA_cases = np.array(CA_cases)
CA_POP = np.array(CA_POP)
CA_AREA = np.array(CA_AREA)
CA_pop = np.array(CA_pop)
CA_pop_area = np.array(CA_pop_area)
# TX
TX = np.zeros(days)
TX_cases = []
TX_POP = []
TX_AREA = []
TX_pop= []
TX_pop_area = []

for i in range(len(tx)):
    tx[i].to_numpy()
for i in range(len(tx)):
    for day in range(days):
        d = day + 11
        TX[day] += tx[i][d]
for day in range(days):
    if TX[day] != 0:
        TX_cases.append(TX[day])
        TX_POP.append(tx_pop)
        TX_AREA.append(tx_area)
        TX_pop.append(TX[day] / tx_pop)
        TX_pop_area.append(TX[day] / tx_area)

TX_cases = np.array(TX_cases)
TX_POP = np.array(TX_POP)
TX_AREA = np.array(TX_AREA)
TX_pop = np.array(TX_pop)
TX_pop_area = np.array(TX_pop_area)
# WA
WA = np.zeros(days)
WA_cases = []
WA_POP = []
WA_AREA = []
WA_pop= []
WA_pop_area = []

for i in range(len(wa)):
    wa[i].to_numpy()
for i in range(len(wa)):
    for day in range(days):
        d = day + 11
        WA[day] += wa[i][d]
for day in range(days):
    if WA[day] != 0:
        WA_cases.append(WA[day])
        WA_POP.append(wa_pop)
        WA_AREA.append(wa_area)
        WA_pop.append(WA[day] / wa_pop)
        WA_pop_area.append(WA[day] / wa_area)

WA_cases = np.array(WA_cases)
WA_POP = np.array(WA_POP)
WA_AREA = np.array(WA_AREA)
WA_pop = np.array(WA_pop)
WA_pop_area = np.array(WA_pop_area)
# FL
FL = np.zeros(days)
FL_cases = []
FL_POP = []
FL_AREA = []
FL_pop= []
FL_pop_area = []

for i in range(len(fl)):
    fl[i].to_numpy()
for i in range(len(fl)):
    for day in range(days):
        d = day + 11
        FL[day] += fl[i][d]
for day in range(days):
    if FL[day] != 0:
        FL_cases.append(FL[day])
        FL_POP.append(fl_pop)
        FL_AREA.append(fl_area)
        FL_pop.append(FL[day] / fl_pop)
        FL_pop_area.append(FL[day] / fl_area)

FL_cases = np.array(FL_cases)
FL_POP = np.array(FL_POP)
FL_AREA = np.array(FL_AREA)
FL_pop = np.array(FL_pop)
FL_pop_area = np.array(FL_pop_area)
# MA
MA = np.zeros(days)
MA_cases = []
MA_POP = []
MA_AREA = []
MA_pop= []
MA_pop_area = []

for i in range(len(ma)):
    ma[i].to_numpy()
for i in range(len(ma)):
    for day in range(days):
        d = day + 11
        MA[day] += ma[i][d]
for day in range(days):
    if MA[day] != 0:
        MA_cases.append(MA[day])
        MA_POP.append(ma_pop)
        MA_AREA.append(ma_area)
        MA_pop.append(MA[day] / ma_pop)
        MA_pop_area.append(MA[day] / ma_area)

MA_cases = np.array(MA_cases)
MA_POP = np.array(MA_POP)
MA_AREA = np.array(MA_AREA)
MA_pop = np.array(MA_pop)
MA_pop_area = np.array(MA_pop_area)
'''
D = Days.reshape((-1, 1))
NJ_model = LinearRegression().fit(D, NJ)
print('NJ Slope:', NJ_model.coef_)
NY_model = LinearRegression().fit(D, NY)
print('NY Slope:', NY_model.coef_)
CA_model = LinearRegression().fit(D, CA)
print('CA Slope:', CA_model.coef_)
TX_model = LinearRegression().fit(D, TX)
print('TX Slope:', TX_model.coef_)
WA_model = LinearRegression().fit(D, WA)
print('WA Slope:', WA_model.coef_)
FL_model = LinearRegression().fit(D, FL)
print('FL Slope:', FL_model.coef_)
MA_model = LinearRegression().fit(D, MA)
print('MA Slope:', MA_model.coef_)

d_future = np.linspace(start = 91, stop = (91+5), num = 5, dtype = int).reshape((-1, 1))
NJ_future = NJ_model.predict(d_future)
NY_future = NY_model.predict(d_future)
CA_future = CA_model.predict(d_future)
TX_future = TX_model.predict(d_future)
WA_future = WA_model.predict(d_future)
FL_future = FL_model.predict(d_future)
MA_future = MA_model.predict(d_future)
'''
########################################
# convert to pandas and write to csv
########################################
# Number of Cases per state
path = 'Data/'

NJDays = np.linspace(start = 1, stop = NJ_cases.shape[0], num = NJ_cases.shape[0])
NYDays = np.linspace(start = 1, stop = NY_cases.shape[0], num = NY_cases.shape[0])
CADays = np.linspace(start = 1, stop = CA_cases.shape[0], num = CA_cases.shape[0])
TXDays = np.linspace(start = 1, stop = TX_cases.shape[0], num = TX_cases.shape[0])
WADays = np.linspace(start = 1, stop = WA_cases.shape[0], num = WA_cases.shape[0])
FLDays = np.linspace(start = 1, stop = FL_cases.shape[0], num = FL_cases.shape[0])
MADays = np.linspace(start = 1, stop = MA_cases.shape[0], num = MA_cases.shape[0])

filename = path+'NJ'+'.pkl'
NJ_write = []
NJ_write.append(NJDays)
NJ_write.append(NJ_cases)
NJ_write.append(NJ_POP)
NJ_write.append(NJ_AREA)
NJ_write.append(NJ_pop)
NJ_write.append(NJ_pop_area)
fileObject = open(filename, 'wb')
pkl.dump(NJ_write, fileObject)
fileObject.close()

filename = path+'NY'+'.pkl'
NY_write = []
NY_write.append(NYDays)
NY_write.append(NY_cases)
NY_write.append(NY_POP)
NY_write.append(NY_AREA)
NY_write.append(NY_pop)
NY_write.append(NY_pop_area)
fileObject = open(filename, 'wb')
pkl.dump(NY_write, fileObject)
fileObject.close()

filename = path+'CA'+'.pkl'
CA_write = []
CA_write.append(CADays)
CA_write.append(CA_cases)
CA_write.append(CA_POP)
CA_write.append(CA_AREA)
CA_write.append(CA_pop)
CA_write.append(CA_pop_area)
fileObject = open(filename, 'wb')
pkl.dump(CA_write, fileObject)
fileObject.close()

filename = path+'TX'+'.pkl'
TX_write = []
TX_write.append(TXDays)
TX_write.append(TX_cases)
TX_write.append(TX_POP)
TX_write.append(TX_AREA)
TX_write.append(TX_pop)
TX_write.append(TX_pop_area)
fileObject = open(filename, 'wb')
pkl.dump(TX_write, fileObject)
fileObject.close()

filename = path+'WA'+'.pkl'
WA_write = []
WA_write.append(WADays)
WA_write.append(WA_cases)
WA_write.append(WA_POP)
WA_write.append(WA_AREA)
WA_write.append(WA_pop)
WA_write.append(WA_pop_area)
fileObject = open(filename, 'wb')
pkl.dump(WA_write, fileObject)
fileObject.close()

filename = path+'FL'+'.pkl'
FL_write = []
FL_write.append(FLDays)
FL_write.append(FL_cases)
FL_write.append(FL_POP)
FL_write.append(FL_AREA)
FL_write.append(FL_pop)
FL_write.append(FL_pop_area)
fileObject = open(filename, 'wb')
pkl.dump(FL_write, fileObject)
fileObject.close()

filename = path+'MA'+'.pkl'
MA_write = []
MA_write.append(MADays)
MA_write.append(MA_cases)
MA_write.append(MA_POP)
MA_write.append(MA_AREA)
MA_write.append(MA_pop)
MA_write.append(MA_pop_area)
fileObject = open(filename, 'wb')
pkl.dump(MA_write, fileObject)
fileObject.close()


plt.title('Number of Confirmed Cases since January 22, 2020')
plt.plot(Days, NJ, color = 'b', label = 'NJ')
plt.plot(Days, NY, color = 'brown', label = 'NY')
plt.plot(Days, CA, color = 'g', label = 'CA')
plt.plot(Days, TX, color = 'orange', label = 'TX')
plt.plot(Days, WA, color = 'purple', label = 'WA')
plt.plot(Days, FL, color = 'k', label = 'FL')
plt.plot(Days, MA, color = 'y', label = 'MA')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Number of Confirmed Cases')
plt.savefig('Figures/Number_of_Cases.png')
'''
plt.plot(d_future, NJ_future, color = 'r', label = 'Prediction')
plt.plot(d_future, NY_future, color = 'r')
plt.plot(d_future, CA_future, color = 'r')
plt.plot(d_future, TX_future, color = 'r')
plt.plot(d_future, WA_future, color = 'r')
plt.plot(d_future, FL_future, color = 'r')
plt.plot(d_future, MA_future, color = 'r')
plt.legend()
plt.savefig('Figures/Number_of_Cases_with_predictions.png')
'''

plt.clf()
plt.title('Cases per square mile')
plt.plot(NJDays, NJ_pop_area, color = 'b', label = 'NJ')
plt.plot(NYDays, NY_pop_area, color = 'brown', label = 'NY')
plt.plot(CADays, CA_pop_area, color = 'g', label = 'CA')
plt.plot(TXDays, TX_pop_area, color = 'orange', label = 'TX')
plt.plot(WADays, WA_pop_area, color = 'purple', label = 'WA')
plt.plot(FLDays, FL_pop_area, color = 'k', label = 'FL')
plt.plot(MADays, MA_pop_area, color = 'y', label = 'MA')
plt.legend()
plt.xlabel('Days since first case')
plt.ylabel('Number of Confirmed Cases per Square Mile')
plt.savefig('Figures/Number_of_Cases_function_of_population_density_from_first.png')

plt.clf()
plt.title('Number of Confirmed Cases as a function of Population\n since First Case')
plt.plot(NJDays, NJ_pop, color = 'b', label = 'NJ')
plt.plot(NYDays, NY_pop, color = 'brown', label = 'NY')
plt.plot(CADays, CA_pop, color = 'g', label = 'CA')
plt.plot(TXDays, TX_pop, color = 'orange', label = 'TX')
plt.plot(WADays, WA_pop, color = 'purple', label = 'WA')
plt.plot(FLDays, FL_pop, color = 'k', label = 'FL')
plt.plot(MADays, MA_pop, color = 'y', label = 'MA')
plt.legend()
plt.xlabel('Days since first case')
plt.ylabel('Percentage of Population Confirmed')
plt.savefig('Figures/Number_of_Cases_function_of_population_from_first.png')


plt.clf()
plt.title('Number of Confirmed Cases\n since First Case')
plt.plot(NJDays, NJ_cases, color = 'b', label = 'NJ')
plt.plot(NYDays, NY_cases, color = 'brown', label = 'NY')
plt.plot(CADays, CA_cases, color = 'g', label = 'CA')
plt.plot(TXDays, TX_cases, color = 'orange', label = 'TX')
plt.plot(WADays, WA_cases, color = 'purple', label = 'WA')
plt.plot(FLDays, FL_cases, color = 'k', label = 'FL')
plt.plot(MADays, MA_cases, color = 'y', label = 'MA')
plt.legend()
plt.xlabel('Days since first case')
plt.ylabel('Number of Cases')
plt.savefig('Figures/Number_of_Cases_from_first.png')



'''
plt.clf()
plt.title('Number of Confirmed Cases as a function of Population Density\n since January 22, 2020')
plt.plot(NJDays, NJ_dense, color = 'b', label = 'NJ')
plt.plot(Days, NY_dense, color = 'brown', label = 'NY')
plt.plot(Days, CA_dense, color = 'g', label = 'CA')
plt.plot(Days, TX_dense, color = 'orange', label = 'TX')
plt.plot(Days, WA_dense, color = 'purple', label = 'WA')
plt.plot(Days, FL_dense, color = 'k', label = 'FL')
plt.plot(Days, MA_dense, color = 'y', label = 'MA')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Percentage of Population Confirmed')
plt.savefig('Figures/Number_of_Cases_function_of_population_density.png')
'''
