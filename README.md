

## Intermediate processing codes

| Code                       | Short description                                               | Input Data                              | Output Data                                          | 
|----------------------------|-----------------------------------------------------------------|-----------------------------------------|------------------------------------------------------|
| comp_radar_dist_params.py  | Computes GEV and covariance by sampling over max rain in events | Radar event file                        | GEV distribution parameters, covariances and others. |
| comp_cpm_events.py         | Compute events from CPM                                         | CPM seasonal max data                   | CPM events and characteristics                       |
| comp_gev_fits.py           | Fit GEV to CPM events.                                          | CPM seasonal maxima, various timeseires | GEV fits and covariates to the data.                 |
| comp_radar_events.py       | Compute events from Radar                                       | Radar seasonal max data                 | Radar events and characteristics                     |
 

## Images used in paper and  code  to generate them.

| Figure Number | Image                      | Code                                | Notes               |
|:--------------|----------------------------|-------------------------------------|:--------------------|
| 1             | carmont_geog_group         | plot_carmont_geog_group.py          | Requires OS API Key |
| 2             | radar_carmont              | plot_radar_carmont.py               |                     |
| 3             | radar_jja                  | plot_radar_jja.py                   |                     |
| 4             | scatter                    | plot_scatter.py                     |                     |
| 5             | cpm_intensity_delta        | plot_cpm_intensity_delta.py         |                     |
| 6             | radar_return_prds          | plot_radar_return_prds.py           |                     |
| 7             | map_return_prds            | plot_radar_return_prds.py           |                     |
| 8             | kde_smooth_events          | plot_kde_smooth_events.py           |                     |
| 9             | intens_prob_ratios         | plot_intens_prob_ratios.py          |                     |
| 10            | kde_smooth_events_2065_2080| plot_kde_smooth_events_2065_2080.py |                | 