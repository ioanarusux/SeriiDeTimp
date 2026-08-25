# serii-de-timp

(proiect de grup)
Previzionarea cursului EUR/RON si legatura cu inflatia si rata ROBOR 3M in Romania

CE ACOPERA CODUL:

UNIVARIATE (pe EUR_RON):
  [1] Trend determinist + descompunere (trend/sezon)
  [2] Test ADF – stationaritate
  [3] Holt-Winters (netezire exponentiala)
  [4] ARIMA/SARIMA (grid simplu, AIC)
  [5] Train/test + prognoza cu interval incredere 95%
  [6] Comparatie SARIMA vs Holt-Winters (RMSE, MAPE)

MULTIVARIATE (EUR_RON, inflatie, ROBOR3M):
  [7] ADF, Johansen, VAR/VECM, Granger, IRF, FEVD
