import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

cmap_z = 'cividis'
cmap_t = 'RdYlBu_r'
cmap_diff = 'bwr'
cmap_error = 'BrBG'

def imcol(ax, fig, data, title='', **kwargs):
    if not 'vmin' in kwargs.keys():
        mx = np.abs(data.max().values)
        kwargs['vmin'] = -mx; kwargs['vmax'] = mx
#     I = ax.imshow(data, origin='lower',  **kwargs)
    I = data.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, add_labels=False, 
                  rasterized=True, **kwargs)
    cb = fig.colorbar(I, ax=ax, orientation='horizontal', pad=0.01, shrink=0.90)
    ax.set_title(title)
    ax.coastlines(alpha=0.5)

def plot_increment(valid, xb_iter, fc_iter):
    fig, axs = plt.subplots(4, 5, figsize=(36, 24), subplot_kw={'projection': ccrs.PlateCarree()})

    for iax, var, cmap, r, t in zip(
        [0], ['z'], [cmap_z], [[47000, 58000]], [r'Z500 [m$^2$ s$^{-2}$]']):
        imcol(axs[iax,0],
                fig, 
                valid[var].isel(time=12), cmap=cmap, 
                vmin=r[0], vmax=r[1], title=f'ERA5 {t} t=12h')
        imcol(axs[iax,1],
                fig, 
                xb_iter[var].isel(time=0).sel(lead_time=12), cmap=cmap, 
                vmin=r[0], vmax=r[1], title=f'Background {t} t=12h')
        imcol(axs[iax,2],
                fig,
                xb_iter[var].isel(time=0).sel(lead_time=12)-valid[var].isel(time=12), cmap=cmap_diff, 
                title=f'Background - ERA5 {t} t=12h')
        imcol(axs[iax,3],
                fig, 
                fc_iter[var].isel(time=0).sel(lead_time=12), cmap=cmap, 
                vmin=r[0], vmax=r[1], title=f'Analysis {t} t=12h')
        imcol(axs[iax,4],
                fig,
                fc_iter[var].isel(time=0).sel(lead_time=12)-xb_iter[var].isel(time=0).sel(lead_time=12), cmap=cmap_diff, 
                title=f'Increment {t} 12h')

    for iax, var, cmap, r, t in zip(
        [1], ['z'], [cmap_z], [[47000, 58000]], [r'Z500 [m$^2$ s$^{-2}$]']):
        imcol(axs[iax,0],
                fig, 
                valid[var].isel(time=24), cmap=cmap, 
                vmin=r[0], vmax=r[1], title=f'ERA5 {t} t=24h')
        imcol(axs[iax,1],
                fig, 
                xb_iter[var].isel(time=0).sel(lead_time=24), cmap=cmap, 
                vmin=r[0], vmax=r[1], title=f'Background {t} t=24h')
        imcol(axs[iax,2],
                fig,
                xb_iter[var].isel(time=0).sel(lead_time=24)-valid[var].isel(time=24), cmap=cmap_diff, 
                title=f'Background - ERA5 {t} t=24h')
        imcol(axs[iax,3],
                fig, 
                fc_iter[var].isel(time=0).sel(lead_time=24), cmap=cmap, 
                vmin=r[0], vmax=r[1], title=f'Analysis {t} t=24h')
        imcol(axs[iax,4],
                fig, 
                fc_iter[var].isel(time=0).sel(lead_time=24)-xb_iter[var].isel(time=0).sel(lead_time=24), cmap=cmap_diff, 
                title=f'Increment {t} 24h')

    for iax, var, cmap, r, t in zip(
        [2], ['z'], [cmap_z], [[47000, 58000]], [r'Z500 [m$^2$ s$^{-2}$]']):
        imcol(axs[iax,0],
                fig, 
                valid[var].isel(time=48), cmap=cmap, 
                vmin=r[0], vmax=r[1], title=f'ERA5 {t} t=48h')
        imcol(axs[iax,1],
                fig, 
                xb_iter[var].isel(time=0).sel(lead_time=48), cmap=cmap, 
                vmin=r[0], vmax=r[1], title=f'Background {t} t=48h')
        imcol(axs[iax,2],
                fig,
                xb_iter[var].isel(time=0).sel(lead_time=48)-valid[var].isel(time=48), cmap=cmap_diff, 
                title=f'Background - ERA5 {t} t=48h')
        imcol(axs[iax,3],
                fig, 
                fc_iter[var].isel(time=0).sel(lead_time=48), cmap=cmap, 
                vmin=r[0], vmax=r[1], title=f'Analysis {t} t=48h')
        imcol(axs[iax,4],
                fig, 
                fc_iter[var].isel(time=0).sel(lead_time=48)-xb_iter[var].isel(time=0).sel(lead_time=48), cmap=cmap_diff, 
                title=f'Increment {t} 48h')

    for iax, var, cmap, r, t in zip(
        [3], ['z'], [cmap_z], [[47000, 58000]], [r'Z500 [m$^2$ s$^{-2}$]']):
        imcol(axs[iax,0],
                fig, 
                valid[var].isel(time=96), cmap=cmap, 
                vmin=r[0], vmax=r[1], title=f'ERA5 {t} t=96h')
        imcol(axs[iax,1],
                fig, 
                xb_iter[var].isel(time=0).sel(lead_time=96), cmap=cmap, 
                vmin=r[0], vmax=r[1], title=f'Background {t} t=96h')
        imcol(axs[iax,2],
                fig,
                xb_iter[var].isel(time=0).sel(lead_time=96)-valid[var].isel(time=96), cmap=cmap_diff, 
                title=f'Background - ERA5 {t} t=96h')
        imcol(axs[iax,3],
                fig, 
                fc_iter[var].isel(time=0).sel(lead_time=96), cmap=cmap, 
                vmin=r[0], vmax=r[1], title=f'Analysis {t} t=96h')
        imcol(axs[iax,4],
                fig, 
                fc_iter[var].isel(time=0).sel(lead_time=96)-xb_iter[var].isel(time=0).sel(lead_time=96), cmap=cmap_diff, 
                title=f'Increment {t} 96h')