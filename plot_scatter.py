# plot scatters between CET and Regional Temp and CET & regional SVP (Pa)

import pathlib
import CPM_rainlib
import CPMlib
import xarray
import commonLib
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
def get_jja_ts(path: pathlib.Path) -> xarray.Dataset:
    """
    Get the JJA time series from monthly-mean data in netcdf file
    :param path: path to data
    :return: dataset containing JJA data
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    ts = xarray.open_dataset(path)
    ts = ts.resample(time='QS-DEC').mean()
    ts = ts.where(ts.time.dt.season == 'JJA', drop=True)
    return ts

def regress(x:xarray.DataArray,y:xarray.DataArray):
    # compute fit using statsmodels.
    X = x.values.flatten()
    X = np.column_stack((X, X ** 2))
    X = sm.add_constant(X)
    Y = y.values.flatten()
    model = sm.OLS(Y, X)
    fit = model.fit()
    return fit



ts_dir = CPM_rainlib.dataDir / 'CPM_ts'
sim_cet = get_jja_ts(ts_dir / 'cet_tas.nc').tas
sim_reg_es = get_jja_ts(ts_dir / 'rgn_svp.nc').tas
sim_reg_tas = get_jja_ts(ts_dir / 'reg_tas.nc').tas
sim_reg_pr = get_jja_ts(ts_dir / 'reg_pr.nc').pr

# now compute the regression coefficient.
fit_es = regress(sim_cet, sim_reg_es)
fit_reg = regress(sim_cet, sim_reg_tas)
fit_pr = regress(sim_cet, sim_reg_pr)
fit_es_reg = regress(sim_reg_tas, sim_reg_es)
# compute today's values
obs_cet = commonLib.read_cet()  # read in the obs CET
obs_cet_jja = obs_cet.where(obs_cet.time.dt.season == 'JJA',drop=True)
t_today = obs_cet_jja.sel(**CPMlib.today_sel).mean()
t_PI = obs_cet_jja.sel(**CPMlib.PI_sel).mean()
# put a scatter plot of CET "now" vs local temp "now"
mn_cet = sim_cet.sel(**CPMlib.today_sel).mean('time')

label=commonLib.plotLabel()
fig_scatter, axis = plt.subplots(nrows=2, ncols=2, num='scatter', clear=True, figsize=[8, 5], layout='constrained')
for var,ax in zip([sim_reg_tas,sim_reg_es,sim_reg_pr],axis.flat):
    ax.scatter(sim_cet, var,color='grey',s=6)
    var_mn = var.sel(**CPMlib.today_sel).mean('time')
    ax.scatter(mn_cet, var_mn, color='red', s=30, marker='+', zorder=100)
    label.plot(ax)
ax=axis[-1][-1]
ax.scatter(sim_reg_tas, sim_reg_es, color='grey', s=6)
mn_sim_reg_tas = sim_reg_tas.sel(**CPMlib.today_sel).mean('time')
mn_sim_reg_es = sim_reg_es.sel(**CPMlib.today_sel).mean('time')
ax.scatter(mn_sim_reg_tas,mn_sim_reg_es, color='red', s=30, marker='+', zorder=100)
label.plot(ax)
# plot best fit lines...
npts = 50


for fit, ax,title in zip([fit_reg, fit_es,fit_pr,fit_es_reg], axis.flat,
                         ["CET vs Reg. Temp", "CET vs Reg. SVP", "CET vs Reg. Precip","Reg T vs Reg SVP"]):
    if title.startswith('CET'):
        x = np.linspace(12, 24, num=npts)
        units='K CET'
        ref_t = t_today
        ref_p1k= t_today+1
        ref_pi = t_PI
    else:
        x=np.linspace(9.5, 19.5, num=npts)
        units = 'K Reg. T'
        ref_pi = fit_reg.predict([[1.0, t_PI, t_PI ** 2]])[0]
        ref_t = fit_reg.predict([[1.0, t_today, t_today ** 2]])[0]
        ref_p1k = fit_reg.predict([[1.0, t_today+1, (t_today+1) ** 2]])[0]
    x = np.column_stack((x, x ** 2))
    best_fit = fit.predict(sm.add_constant(x))
    ax.plot(x[:,0], best_fit, color='black', linewidth=3)
    fit_PI = fit.predict([[1.0, ref_pi, ref_pi ** 2]])[0]
    fit_today = fit.predict([[1.0,ref_t,ref_t**2]])[0]
    fit_p1k = fit.predict([[1.0, ref_p1k, ref_p1k ** 2]])[0]
    ax.plot(ref_t,fit_today,marker='h',ms=10,color='red')
    ax.plot(ref_pi, fit_PI, marker='h', ms=10, color='green')
    # work out fractional change per K CET increase
    if 'Temp' in title:
        fract_increase = (fit_p1k-fit_today)
        err = fit.bse[1]
        text = f"{fract_increase:4.2f} $\pm$ {err:4.2f} R$^2$:{fit.rsquared*100:3.0f} %"
    else:
        fract_increase = 100*((fit_p1k-fit_today) / fit_today)
        err = 100*(fit.bse[1]/ fit_today)
        text = f"{fract_increase:4.2f} $\pm$ {err:4.2f} R$^2$:{fit.rsquared*100:3.0f} %"
    ax.text(0.05, 0.8,text,ha='left',va='bottom',backgroundcolor='grey',transform=ax.transAxes)
    print(f"Change per {units} = {fract_increase:4.1f}, Err: {err:4.2f} (%): {fit.rsquared*100:3.0f} %")
    ax.set_title(title)



for a,ylabel in zip(axis.flat[0:3],[r"Temp ($^\circ$C)",'SVP (Pa)','Precip (mm/day)']):
    a.set_xlabel(r"CET ($^\circ$C)")
    a.set_ylabel(ylabel)
axis.flat[-1].set_xlabel(r"Regional Temp ($^\circ$C)")
axis.flat[-1].set_ylabel(r"SVP (Pa)")

fig_scatter.show()
commonLib.saveFig(fig_scatter,figtype=['pdf','png'])
