from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import datetime
from dateutil.parser import parse
import re
from cvxopt import matrix,solvers


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

data_all=pd.read_excel('data_all0829.xls',index_col='TradingDay')
df_daily_return=data_all['2013-07-01':]
assets=df_daily_return.columns
df_vol=pd.DataFrame(df_daily_return.std()*np.sqrt(250)).T
annualizedReturn=df_daily_return.mean()*250
df_covmat=df_daily_return.cov()*250
df_corr=df_daily_return.corr()
# 风险厌恶系数
delta = 2
# 观点权重tau
tau = 0.03     


#生成自己的观点

#观点1：美国标普500指数年化收益率设为6%（在原来11%的基础上下调）

#观点2：香港恒生指数年化收益率为5%（相比原来平均3.4%小幅上调）

#观点3：沪深300指数收益率比中小板指数收益率高4%（相比原来2%小幅上调）

#当然，这样获得的观点不一定正确，也比较片面，实际考虑的因素可以很多。这里只做个简单分析。

#然后根据观点得到 P Q Ω

P = np.array([[0,0,0,1,0],[0,0,1,0,0],[1,-1,0,0,0]])
print "P:",P
Q = np.array([0.06,0.05,0.04])
print "Q:",Q
Omega = tau*(P.dot(df_daily_return.cov()*250).dot(P.transpose()))
Omega = np.diag(np.diag(Omega,k=0))
print "Omega:",Omega



#计算后验收益及方差

adjustedReturn = annualizedReturn + tau * df_covmat.dot(
    P.transpose()).dot(np.linalg.inv(Omega + tau * (
    P.dot(df_covmat).dot(P.transpose())))).dot(Q - P.dot(annualizedReturn))
right = (tau)* df_covmat.dot(P.transpose()).dot(np.linalg.inv(
    Omega + P.dot(df_covmat).dot(P.transpose()))).dot(P.dot(tau*df_covmat))
right =  right.transpose()
right = right.set_index(annualizedReturn.index)
M = tau*df_covmat - right
Sigma_p = df_covmat + M
#adjustedReturn = adjustedReturn.as_matrix()
Sigma_p = Sigma_p.as_matrix()



plt.figure(figsize=(15,8))
annualizedReturn.plot(label='originalReturn')
adjustedReturn.plot(label='adjustedReturn')
plt.legend(loc=0)
plt.grid(True)

#权重对比
#val_ls = [np.random.randint(100) + i*20 for i in range(7)]
scale_ls = range(5)
index_ls = ['HS300','Small_Medium','HSIndex','SP500','Gold']
#plt.bar(scale_ls, val_ls)
plt.xticks(scale_ls,index_ls)  ## 可以设置坐标字
plt.title('Return Comparison')


port_returns = []
port_variance = []
for p in range(1000):
    weights = np.random.random(5)
    weights /= np.sum(weights)
    port_returns.append(np.sum(adjustedReturn*weights))
    port_variance.append(np.sqrt(np.dot(weights.T, np.dot(df_covmat, weights))))

port_returns = np.array(port_returns)
port_variance = np.array(port_variance)

#无风险利率设定为2%
risk_free = 0.02
plt.figure(figsize = (12,8))
plt.scatter(port_variance, port_returns, c=(port_returns-risk_free)/port_variance, marker = 'o')
plt.grid(True)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label = 'Sharpe Ratio')


#画出有效前沿
risks, returns, weights = [], [], []
for level_return in np.linspace(min(adjustedReturn), max(adjustedReturn),1000):
    P = 2 * Sigma_p
    q = matrix(np.zeros(5))

    G = matrix(np.diag(-1 * np.ones(5)))
        
    h = matrix([0.0,0.0,0.0,0.0,0.0], (5, 1))    
    A = matrix(np.row_stack((np.ones(5), adjustedReturn)))
    b = matrix([1.0, level_return])
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    risks.append(np.sqrt(sol['primal objective']))
    returns.append(level_return)
    weights.append(dict(zip(assets, list(sol['x'].T))))
    
output = {"returns": returns,
         "risks": risks,
         "weights": weights}
output = pd.DataFrame(output)

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111)
ax.plot(output['risks'], output['returns'])
ax.set_title('Efficient Frontier', fontsize=14)
ax.set_xlabel('Standard Deviation', fontsize=12)
ax.set_ylabel('Expected Return', fontsize=12)
ax.tick_params(labelsize=12)
plt.show()

#在无约束的条件下，求出各自的在最大效用函数下的资产配置权重w1和w_adj
w1 = np.linalg.inv(delta*df_covmat).dot(annualizedReturn)
w1 = pd.Series(w1,index=annualizedReturn.index)
w_adj = np.linalg.inv(delta*Sigma_p).dot(adjustedReturn)
w_adj = pd.Series(w_adj,index=adjustedReturn.index)
plt.figure(figsize = (12,8))
w1.plot(label='origin weight')
w_adj.plot(label='adjusted weight')
plt.legend(loc=0)
scale_ls = range(5)
index_ls = ['HS300','Small_Medium','HSIndex','SP500','Gold']
#plt.bar(scale_ls, val_ls)
plt.xticks(scale_ls,index_ls)  ## 可以设置坐标字
plt.title('Weight Comparison')

#在有约束的条件下，求出各自的在最大效用函数下的资产配置权重w1和w_adj

P = delta * matrix(df_covmat.as_matrix())
q = matrix(-annualizedReturn.T)
G = matrix(np.diag(-1 * np.ones(5)))
h = matrix(0., (5, 1))
A = matrix(np.ones(5)).T
b = matrix([1.0])
solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h, A, b)
w1 = np.array(sol['x'].T)[0]
w1 = pd.Series(w1,index=annualizedReturn.index)

P = delta * matrix(Sigma_p)
q = matrix(-adjustedReturn.T)
G = matrix(np.diag(-1 * np.ones(5)))
h = matrix(0., (5, 1))
A = matrix(np.ones(5)).T
b = matrix([1.0])
solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h, A, b)
w_adj = np.array(sol['x'].T)[0]
w_adj = pd.Series(w_adj,index=adjustedReturn.index)


plt.figure(figsize = (12,8))
w1.plot(label='origin weight')
w_adj.plot(label='adjusted weight')
plt.legend(loc=0)
scale_ls = range(5)
index_ls = ['HS300','Small_Medium','HSIndex','SP500','Gold']
#plt.bar(scale_ls, val_ls)
plt.xticks(scale_ls,index_ls)  ## 可以设置坐标字
plt.title('Weight Comparison')
