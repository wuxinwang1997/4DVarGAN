import numpy as np
from pysteps.utils.spectral import rapsd, corrcoef
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xarray as xr
from scipy import ndimage
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
# import scienceplots
# plt.style.use(['science','no-latex'])

def compute_mean_spectral_density(data: xr.DataArray, timestamp=None, num_times=None):
    # data = data.sel(time=slice(self.time_period[0], self.time_period[1]))
    # data = data.isel(lon=slice(0 + 10, 59 + 10), lat=slice(0 + 10, 59 + 10))
    num_frequencies = np.max((len(data.lat.values),
                              len(data.lon.values))) / 2
    mean_spectral_density = np.zeros(int(num_frequencies))
    if num_times is None:
        num_times = int(len(data.time))

    elif timestamp is not None:
        num_times = 1

    else:
        num_times = num_times

    for t in range(num_times):
        if timestamp is not None:
            tmp = data.sel(time=timestamp)['z'].values
        else:
            tmp = data.isel(time=t)['z'].values
        psd, freq = rapsd(tmp, return_freq=True, normalize=True, fft_method=np.fft)
        mean_spectral_density += psd
    mean_spectral_density /= num_times

    return mean_spectral_density, freq

def cvae_compute_mean_spectral_density(data: xr.DataArray, timestamp=None, num_times=None):
    # data = data.sel(time=slice(self.time_period[0], self.time_period[1]))
    # data = data.isel(lon=slice(0 + 10, 59 + 10), lat=slice(0 + 10, 59 + 10))
    num_frequencies = np.max((len(data.lat.values), len(data.lon.values))) / 2

    mean_spectral_density = np.zeros(int(num_frequencies))

    if num_times is None:
        num_times = int(len(data.init_time))
    elif timestamp is not None:
        num_times = 1

    else:
        num_times = num_times

    for t in range(num_times):
        if timestamp is not None:
            tmp = data.sel(init_time=timestamp)['z'].values
        else:
            tmp = data.isel(init_time=t)['z'].values
        psd, freq = rapsd(tmp, return_freq=True, normalize=True, fft_method=np.fft)
        mean_spectral_density += psd
    mean_spectral_density /= num_times

    return mean_spectral_density, freq

def plot_psd(era5_psd, var4d_psd, map_psd, cgan_psd, freq, obs_partial, axis=None, fname=None, fontsize=None, linewidth=None):

    if axis is None:
        _, ax = plt.subplots(figsize=(7, 6))
    else:
        ax = axis

    plt.rcParams.update({'font.size': 12})
    x_vals = 1 / freq * 5.625 * 111 / 2

    ax.plot(x_vals, era5_psd, label='ERA5', color='k', linewidth=linewidth)

    for i in np.arange(len(obs_partial)):
        ax.plot(x_vals, var4d_psd[i], label=f'4DVar {int(100*obs_partial[i])}% Obs', color='b', linewidth=linewidth, linestyle='-.', alpha=min(1, 2*np.sqrt(obs_partial[i])))
        ax.plot(x_vals, map_psd[i], label=f'DirectMap {int(100*obs_partial[i])}% Obs', color='g', linewidth=linewidth, linestyle='-.', alpha=min(1, 2*np.sqrt(0.1+obs_partial[i])))
        ax.plot(x_vals, cgan_psd[i], label=f'CGAN {int(100*obs_partial[i])}% Obs', color='r', linewidth=linewidth, linestyle='-.', alpha=min(1, 2*np.sqrt(0.1+obs_partial[i])))
    # ax.plot(x_vals, self.quantile_mapping_psd, label='Quantile mapping', color='m', linewidth=linewidth)
    # ax.plot(x_vals, self.gan_psd, label='GAN', color='c', linewidth=linewidth)
    ax.legend(loc='lower left', fontsize=fontsize)
    ax.set_xlim(x_vals[1] + 1024, x_vals[-1] - 32)
    ax.set_yscale('log', base=10)
    ax.set_xscale('log', base=2)
    ax.set_xticks([2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13])
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid()
    # ax.set_ylim(4.5e-5, 0.07)
    ax.set_xlabel(r'WaveLength [km]', fontsize=fontsize)
    ax.set_ylabel('PSD [a.u]', fontsize=fontsize)

    if fname is not None:
        plt.savefig(fname, format='jpg', dpi=300, bbox_inches='tight')

def subplot_psd(psds, names, freq, obs_partial, axis=None, fname=None, fontsize=None, linewidth=None):

    fig = plt.figure(figsize=(12, 12))

    # if axis is None:
    #     _, ax = plt.subplots(figsize=(7, 6))
    # else:
    #     ax = axis

    plt.rcParams.update({'font.size': 12})
    x_vals = 1 / freq * 5.625 * 111 / 2
    x_ticks = [2**8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13]
    for i in range(len(obs_partial)):
        ax = fig.add_subplot(2, 2, i+1)
        for j in range(len(psds)):
            if names[j]=='ERA5':
                ax.plot(x_vals, psds[j], label=f'{names[j]}', linewidth=linewidth, color='k')
            else:
                ax.plot(x_vals, psds[j][i], label=f'{names[j]}', linewidth=linewidth)
        # ax.plot(x_vals, var4d_psd[i], label=f'4DVar', color='b', linewidth=linewidth, linestyle='-.') #, alpha=min(1, 2*np.sqrt(obs_partial[i])))
        # ax.plot(x_vals, map_psd[i], label=f'DirectMap', color='g', linewidth=linewidth, linestyle='-.') #, alpha=min(1, 2*np.sqrt(0.1+obs_partial[i])))
        # ax.plot(x_vals, cgan_psd[i], label=f'CGAN', color='r', linewidth=linewidth, linestyle='-.') #, alpha=min(1, 2*np.sqrt(0.1+obs_partial[i])))
    # ax.plot(x_vals, self.quantile_mapping_psd, label='Quantile mapping', color='m', linewidth=linewidth)
    # ax.plot(x_vals, self.gan_psd, label='GAN', color='c', linewidth=linewidth)
        ax.legend(loc='lower left', fontsize=fontsize)
        ax.set_xlim(x_vals[1] + 1024, x_vals[-1] - 32)
        ax.set_yscale('log', base=10)
        ax.set_xscale('log', base=2)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.grid()
        # ax.set_ylim(4.5e-5, 0.07)
        ax.set_xlabel(r'WaveLength [km]', fontsize=fontsize)
        ax.set_ylabel('PSD [a.u]', fontsize=fontsize)
        plt.title(f'{int(100*obs_partial[i])}% Observations')

        axins = inset_axes(ax, width="40%", height="50%", loc='upper right', bbox_to_anchor=(0.3, 0.1, 0.7, 0.9),
                           bbox_transform=ax.transAxes)
        for j in range(len(psds)):
            axins.plot(x_vals, psds[j] if names[j]=='ERA5' else psds[j][i], label=f'{names[j]}', linewidth=linewidth)
        # axins.plot(x_vals, var4d_psd[i], label=f'4DVar', color='b', linewidth=linewidth,
        #         linestyle='-.')  # , alpha=min(1, 2*np.sqrt(obs_partial[i])))
        # axins.plot(x_vals, map_psd[i], label=f'DirectMap', color='g', linewidth=linewidth,
        #         linestyle='-.')  # , alpha=min(1, 2*np.sqrt(0.1+obs_partial[i])))
        # axins.plot(x_vals, cgan_psd[i], label=f'CGAN', color='r', linewidth=linewidth, linestyle='-.')

        sx = [1406, 384*2, 384*2, 1406, 1406]
        sy = [10e-11, 10e-11, 5*10e-9, 5*10e-9, 10e-11]
        ax.plot(sx, sy, 'black', linewidth=1)

        xy, xy2 = (1406, 5*10e-9), (1406, 5*10e-9)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA='data', coordsB='data', axesA=axins, axesB=ax)
        axins.add_artist(con)

        xy, xy2 = (384*2, 10e-11), (384*2, 10e-11)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA='data', coordsB='data', axesA=axins, axesB=ax)
        axins.add_artist(con)

        axins.set_ylim(10e-11, 5*10e-9)
        axins.set_xlim(1406, 384*2)
        axins.set_yscale('log', base=10)
        axins.set_xscale('log', base=2)
        axins.get_xaxis().set_visible(False)
        axins.get_yaxis().set_visible(False)

    if fname is not None:
        plt.savefig(fname, format='pdf', dpi=300, bbox_inches='tight')

def compute_spatial_variance(data: xr.DataArray, timestamp=None, num_times=None):
    # data = data.sel(time=slice(self.time_period[0], self.time_period[1]))
    # data = data.isel(lon=slice(0 + 10, 59 + 10), lat=slice(0 + 10, 59 + 10))
    num_frequencies = np.max((len(data.lat.values),
                              len(data.lon.values))) / 2
    mean_outVariance = np.zeros(int(num_frequencies))
    if num_times is None:
        num_times = int(len(data.init_time))

    elif timestamp is not None:
        num_times = 1

    else:
        num_times = num_times

    for t in range(num_times):
        if timestamp is not None:
            tmp = data.sel(init_time=timestamp)['z'].values
        else:
            tmp = data.isel(init_time=t)['z'].values
        outVariance = ndimage.generic_filter(tmp, np.var, size=1)
        mean_outVariance += outVariance
    mean_outVariance /= num_times

    return mean_outVariance