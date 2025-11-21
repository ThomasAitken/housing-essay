# Data Analysis for "The Real Causes of Expensive Housing: A Global Perspective"

This is the repo accompanying my Substack post, "The Real Causes of Expensive Housing: A Global Perspective". This is not my finest work as a programmer; it just a rough-and-ready collection of the scripts that I wrote (with AI assistance) to generate the datasets and charts that informed my essay.

I am just open-sourcing in the interests of transparency, and so I can document how I obtained my datasets.

## Notes on Main Datasets: `data_2000-2025.csv` and `data_2010-2025.csv`

A better way of viewing these datasets is on [Google Sheets](https://docs.google.com/spreadsheets/d/1sm-ZR8VECos0gErKGUHBC3uFT7c5id4fr-RSUNHPLnQ/edit?usp=sharing). This data was mostly put together in a painstaking fashion using manual data entry. You will note that I give links to the data sources for each of my raw-data columns. For most of these raw-data sources, it is fairly obvious how to reproduce my data entry, although for some there was some simple calculation involved. These special cases are explained below:

### Real Residential Property Price Growth
As my spreadsheet documents, I obtained this data for every country from the St Louis Fed - specifically their charts, like [this](https://fred.stlouisfed.org/series/QAUR628BIS) that are indexed at 2010. The cumulative growth percent value was arrived at for the 2010-2025 period by taking the Q1 2025 and subtracting 100 (then rounding). The same value was arrived at for the 2000-2025 period by computing `({Q1_2025 - Q1_2000} / {Q1_2000}) * 100` (then rounding).

### Population Growth
This was uniformly obtained from Google Data Commons. At the time I was looking at this data, it only went up to 2024 for each country. I therefore computed the percentage by finding the growth percentage between {start_year} and 2024.


### Change in Dwellings per 1000
I only have this metric for the post-2010 data because there is no OECD data going further back (at least not that I can find). The data for this was fairly straightforward to obtain and calculate (via an Excel formula) for every country except Korea. For Korea, the OECD dataset has Korea's Dwellings per 1000 at a non-credibly low level of 310 in 2013 and no value for the end of the series. I think this comes from relying on Korea's census number of housing units which is for some reason way lower than the data [here](https://www.lh.or.kr/menu.es?mid=a20106000000&utm_source=chatgpt.com).

Here's a snippet from ChatGPT that explains where my own numbers ultimately came from for Korea:

"""
The clean, consistent way (what you probably want)
If we stay entirely in the “number of houses” / supply-ratio series (the one that makes sense for housing-policy work), it looks like this:
```
year,population_m,houses_m,houses_per_1000
2010,48.58,17.739,365.2
2011,49.41,18.082,366.0
2012,50.33,18.414,365.9
2013,50.57,18.742,370.6
2014,50.81,19.161,377.1
2015,51.03,19.559,383.3
2016,51.22,19.877,388.1
2017,51.36,20.313,395.5
2018,51.59,20.818,403.6
2019,51.77,21.310,411.7
2020,51.84,21.674,418.1
2021,51.74,21.917,423.6
```
[Houses 2010–2021: “The number of houses (thousand units)” from LH/KOSIS](https://www.lh.or.kr/menu.es?mid=a20106000000&utm_source=chatgpt.com) 

[Population: KOSTAT / usual national series for those years](https://kostat.go.kr/board.es?act=view&bid=11747&list_no=419981&mid=a20108070000&nPage=1&ref_bid=&tag=&utm_source=chatgpt.com)


### Real Wage Growth
My real wage growth data was computed via the script `./exploring_oecd_real_wages_growth/real_wages_growth_oecd.py` then fed back into this dataset.

### Employment Growth
Like the data on property price growth, this was from the St Louis Fed and I used the same methodology to get a single growth figure for each country.


## Set up
This is an old-fashioned pip-based repo so set up your virtual env then install dependencies with `pip install -r requirements.txt`.

## Scripts

### `generate_regressions.py`
This is the script that generated most of the linear regressions I include in my essay. It is the source of the output for all the charts in `./2000_outputs` (run with `python generate_regressions.py --csv data_2000-2025.csv --outdir 2000_outputs`) and `./2010_outputs` (run with `python generate_regressions.py --csv data_2010-2025.csv --outdir 2010_outputs`).

### `./exploring_oecd_real_wages_growth/real_wages_growth_oecd.py`
This script was used to generate stats on real wages growth in OECD countries since 2000 and since 2010. It pulls raw data then does some dataframe processing and spits out a csv and chart. This script is vibe-coded and I didn't bother to add a CLI so to change the start year you have to overwrite the `START_YEAR` constant.

### `./exploring_change_in_urban_area/global_human_settlement_layer.py`
This one uses data I pulled down from [the Global Human Settlement Layer website](https://human-settlement.emergency.copernicus.eu/GHSLDataTools.php) to try to see if there was anything to be learned about property price growth from urban land expansion. The link to download the .xlsx file is [here](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_GLOBE_R2024A/V1-1/GHS_UCDB_GLOBE_R2024A_V1_1.zip). The link to download the .csv file is [here](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_MTUC_GLOBE_R2024A/V1-1/GHS_UCDB_MTUC_GLOBE_R2024A_V1_1.zip).
