import glob
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from sys import exit
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import gmean
from plotly.graph_objs.layout import XAxis, YAxis
from rich import print
from rich.traceback import install

# install()

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# https://eonlabs.notion.site/ITH-544a79dca6c74d349afb309dcde816e2
# https://www.get-digital-help.com/automate-net-asset-value-nav-calculation-on-your-stock-portfolio-vba-in-excel/

# Directory that holds data for the purposes of calculating their Investment Time Horizon Epoch (ITHE) and their
# related matices Toggle this boolean to control the behavior
delete_everything = False  # Set to False if you want to keep things

output_dir = Path('synth_ithes')

if delete_everything:
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print("Sayonara, files! 🚀")  # Your files have left the chat
    else:
        print("Hmm, nothing to delete here. 🤔")

# Create anew after the great purge
output_dir.mkdir(parents=True, exist_ok=True)
nav_dir = output_dir / "NAV_data"
nav_dir.mkdir(parents=True, exist_ok=True)


def load_config(filename="config.json"):
    """
    config.json file contains the following:
    {
    "api_key": "key_string_here"
    }
    """
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


def generate_synthetic_nav(start_date, end_date, avg_daily_return=0.00010123, daily_return_volatility=0.009, df=4):
    dates = pd.date_range(start_date, end_date)
    # walk = np.random.normal(loc=avg_daily_return, scale=daily_return_volatility, size=len(dates))
    walk = stats.t.rvs(df, loc=avg_daily_return, scale=daily_return_volatility, size=len(dates))
    walk = np.cumsum(walk)
    drawdown = False
    for i in range(len(dates)):
        if drawdown:
            walk[i] -= np.random.uniform(0.001, 0.003)
            if np.random.rand() < 0.02:
                drawdown = False
        elif np.random.rand() < 0.05:
            drawdown = True
    walk = walk - walk[0] + 1  # Normalize the series so that it starts with 1
    nav = pd.DataFrame(data=walk, index=dates, columns=['NAV'])
    nav.index.name = 'Date'
    nav['PnL'] = nav['NAV'].diff()
    nav['PnL'] = nav['PnL'].fillna(nav['NAV'].iloc[0] - 1)  # Adjust the first PnL value accordingly
    return nav


def calculate_sharpe_ratio(returns, rf=0., nperiods=None, annualize=True, trading_year_days=252):
    """
    Determines the Sharpe ratio of a strategy.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    rf : float, optional
        Constant risk-free return throughout the period
    nperiods : int, optional
        Defines the length of a 'period'. Defaults to 252 (the number of
        trading days in a year) for both kind of returns.
        - See full explanation in :func:`~pyfolio.timeseries.aggregate_returns`.
    annualize : bool, optional
        Determines if the statistic will be annualized.
        - See full explanation in :func:`~pyfolio.timeseries.aggregate_returns`.
    trading_year_days: int, optional
        The number of trading days in a year. Defaults to 252.
    Returns
    -------
    float, pd.Series
        depends on input type
        - float : When passed a one-dimensional array.
        - pd.Series : When passed a DataFrame.
    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.
    """
    if len(returns) < 2:
        print("Oh snap! The array is less than 2! Abort mission! 🚨")
        return np.nan

    if np.std(returns, ddof=1) == 0:
        print("Oh snap! The np.std is zero! Abort mission! 🚨")
        return np.nan

    if nperiods is None:
        nperiods = trading_year_days

    returns = returns.copy()
    returns = returns[~np.isnan(returns)]

    if isinstance(returns, pd.Series):
        returns = returns.values

    if annualize:
        return np.sqrt(nperiods) * (np.mean(returns - rf) /
                                    np.std(returns, ddof=1))
    print(f"Returns in calculate_sharpe_ratio: {returns}")
    return np.mean(returns - rf) / np.std(returns, ddof=1)


# Helper function to find the maximum fractional drawdown as the Trailing Minimum Acceptable Excess Gain (TMAEG)
def calculate_max_drawdown(nav_values):
    return np.max(1 - nav_values / np.maximum.accumulate(nav_values))


def add_comment_to_file_finder_column(filename, comment, retries=20, delay=3):
    # Check if the script is running on macOS
    if platform.system() != 'Darwin':
        return
    filename = os.path.abspath(filename)
    script = f'''
    tell application "Finder"
        set filePath to POSIX file "{filename}" as alias
        set currentComments to (get comment of filePath)
        if currentComments is "" or currentComments is "{comment}" then
            set comment of filePath to "{comment}"
        else
            display dialog "The current comment is different from the one to be replaced. Stopping the script."
            error number -128
        end if
        return
    end tell
    '''

    for i in range(retries):
        result = subprocess.run(['osascript', '-e', script])
        if result.returncode == 0:  # Success
            break
        else:
            print(
                f"Attempt {i + 1} to tag file failed with return code {result.returncode}. Retrying after {delay} seconds...")
            time.sleep(delay)  # Wait for a few seconds before trying again
    if result.returncode != 0:  # If tagging failed after all retries
        print("Failed to tag file after all retries. Stopping the script.")
        exit()  # Stop the script


def calculate_deepest_troughs_and_new_highs(nav_values, running_max_nav):
    deepest_troughs_after_new_high = pd.Series(index=nav_values.index, dtype=float)
    new_high_flag = pd.Series(index=nav_values.index, dtype=int)
    current_max = running_max_nav[0]
    current_trough = nav_values[0]
    for i in range(1, len(nav_values)):
        if running_max_nav[i] > current_max:
            current_max = running_max_nav[i]
            current_trough = nav_values[i]
            new_high_flag[i] = 1
        elif nav_values[i] < current_trough:
            current_trough = nav_values[i]
        deepest_troughs_after_new_high[i] = current_trough
    return deepest_troughs_after_new_high, new_high_flag


def calculate_max_dd_points_after_new_high(drawdowns, new_high_flag):
    max_dd_points = pd.Series(np.zeros(len(drawdowns)), index=drawdowns.index)
    current_max_dd = 0
    max_dd_index = -1
    for i in range(1, len(drawdowns)):
        if new_high_flag[i] == 1:
            if max_dd_index != -1:
                max_dd_points[max_dd_index] = current_max_dd
            current_max_dd = 0
            max_dd_index = -1
        else:
            if drawdowns[i] > current_max_dd:
                current_max_dd = drawdowns[i]
                max_dd_index = i
    if max_dd_index != -1:
        max_dd_points[max_dd_index] = current_max_dd
    return max_dd_points


def calculate_geometric_mean_of_drawdown(nav_values):
    running_max_nav = nav_values.cummax()
    deepest_troughs_after_new_high, new_high_flag = calculate_deepest_troughs_and_new_highs(nav_values, running_max_nav)
    drawdowns_to_deepest_troughs = running_max_nav - deepest_troughs_after_new_high
    max_dd_points = calculate_max_dd_points_after_new_high(drawdowns_to_deepest_troughs, new_high_flag)
    max_dd_points_fraction = max_dd_points / running_max_nav
    spike_values = max_dd_points_fraction[max_dd_points_fraction > 0]
    if spike_values.empty:  # Check if spike_values is empty
        geometric_mean = np.nan  # Return NaN or some other appropriate value
    else:
        geometric_mean = gmean(spike_values)
    return geometric_mean


def calculate_excess_gain_excess_loss(hurdle, nav):
    original_df = nav.copy() if isinstance(nav, pd.DataFrame) and 'NAV' in nav.columns else None
    nav = nav['NAV'] if original_df is not None else nav
    excess_gain = excess_loss = 0
    excess_gains = [0]
    excess_losses = [0]
    excess_gains_at_ith_epoch = [0]
    last_reset_state = False
    ith_epochs = [False] * len(nav)
    endorsing_crest = endorsing_nadir = candidate_crest = candidate_nadir = nav.iloc[0]
    for i, (equity, next_equity) in enumerate(zip(nav[:-1], nav[1:])):
        if next_equity > candidate_crest:
            # excess_gain = next_equity / endorsing_crest - 1
            excess_gain = next_equity / endorsing_crest - 1 if endorsing_crest != 0 else 0
            candidate_crest = next_equity
        if next_equity < candidate_nadir:
            excess_loss = 1 - next_equity / endorsing_crest
            candidate_nadir = next_equity
        reset_candidate_nadir_excess_gain_and_excess_loss = excess_gain > abs(
            excess_loss) and excess_gain > hurdle and candidate_crest >= endorsing_crest
        if reset_candidate_nadir_excess_gain_and_excess_loss:
            endorsing_crest = candidate_crest
            endorsing_nadir = candidate_nadir = equity
            excess_gains_at_ith_epoch.append(excess_gain if not last_reset_state else 0)
        else:
            endorsing_nadir = min(endorsing_nadir, equity)
            excess_gains_at_ith_epoch.append(0)
        last_reset_state = reset_candidate_nadir_excess_gain_and_excess_loss
        excess_gains.append(excess_gain)
        excess_losses.append(excess_loss)
        if reset_candidate_nadir_excess_gain_and_excess_loss:
            excess_gain = excess_loss = 0
        # Adjust ith_epochs
        ith_epoch_condition = len(excess_gains) > 1 and excess_gains[-1] > excess_losses[-1] and excess_gains[
            -1] > hurdle
        ith_epochs[i + 1] = ith_epoch_condition  # Adjusted to match original function
    num_of_ith_epochs = ith_epochs.count(True)

    # Calculate the ith_interval_separators and ith_intervals
    ith_interval_separators = [i for i, x in enumerate(ith_epochs) if x]
    ith_interval_separators.insert(0, 0)  # The first point of NAV is always an ith_interval_separator.
    ith_intervals = np.diff(ith_interval_separators)
    ith_intervals_cv = np.std(ith_intervals) / np.mean(ith_intervals)  # Calculate the CV of the ith_intervals

    if original_df is not None:
        original_df['Excess Gains'] = excess_gains
        original_df['Excess Losses'] = excess_losses
        original_df['ITHEs'] = ith_epochs  # Add new column 'ITHEs'
        return original_df
    else:
        return excess_gains, excess_losses, num_of_ith_epochs, ith_epochs, ith_intervals_cv


def get_first_non_zero_digits(num, digit_count):
    # Get first 12 non-zero, non-decimal, non-negative digits
    non_zero_digits = ''.join([i for i in str(num) if i not in ['0', '.', '-']][:12])
    # Then trim or pad to the desired length
    uid = (non_zero_digits + '0' * digit_count)[:digit_count]
    return uid


def calculate_pnl_from_nav(nav_data):
    """
    Calculate PnL from NAV as a fractional percentage.
    """
    nav_data['PnL'] = nav_data['NAV'].diff() / nav_data['NAV'].shift(1)
    nav_data['PnL'].iloc[0] = 0  # First row of PnL is always zero
    return nav_data


def process_nav_data(nav_data, output_dir, qualified_results, nav_dir, TMAEG, instrument=False):
    # Extract the first six non-zero digits from the first two rows of the NAV column
    uid_part1 = get_first_non_zero_digits(nav_data['NAV'].iloc[0], 6)
    uid_part2 = get_first_non_zero_digits(nav_data['NAV'].iloc[1], 6)
    # Concatenate the two parts to form the UID
    uid = uid_part1 + uid_part2
    print(f'{uid=}')

    # Initialize filename with a default value
    filename = None
    ith_durations = None
    excess_losses_at_ithes = None

    if instrument:
        sharpe_ratio = 9999
        average_daily = 9999
    else:
        sharpe_ratio = calculate_sharpe_ratio(nav_data['PnL'].dropna())
        days_elapsed = len(nav_data.resample('D'))
        print("Number of days elapsed:", days_elapsed)

    calculated_nav = calculate_excess_gain_excess_loss(TMAEG, nav_data)
    print(f'{calculated_nav=}')
    ith_epochs = calculated_nav[calculated_nav['ITHEs']].index
    print(f'{ith_epochs=}')
    num_of_ith_epochs = len(ith_epochs)
    print(f'{num_of_ith_epochs=}')

    if sr_lower_bound < sharpe_ratio < sr_upper_bound and ith_epochs_lower_bound < num_of_ith_epochs < ith_epochs_upper_bound:
        print(f"Found {num_of_ith_epochs=}, {sharpe_ratio=}")
        # cumulative_sum = nav_data['NAV'].iloc[-1]
        ith_dates = calculated_nav[calculated_nav['ITHEs']].index
        ith_dates = ith_dates.insert(0, calculated_nav.index[0])
        ith_dates = ith_dates.append(pd.Index([calculated_nav.index[-1]]))
        print(f'fixed: {ith_dates=}')
        ithe_ct = len(ith_dates) - 2
        days_taken_to_ithe = days_elapsed / ithe_ct
        ith_indices = [calculated_nav.index.get_loc(date) for date in ith_dates]
        ith_durations = np.diff(ith_indices)
        print(f'{ith_durations=}')
        ith_cv = np.std(ith_durations) / np.mean(ith_durations)

        # Calculate the coefficient of variation for Excess Losses at ITHEs
        excess_losses_at_ithes = calculated_nav[calculated_nav['ITHEs']]['Excess Losses']
        excess_losses_at_ithes = excess_losses_at_ithes[excess_losses_at_ithes != 0]  # Exclude zero values
        last_excess_loss = calculated_nav['Excess Losses'].iloc[
            -1]  # Include the last value of Excess Losses (even if it is not flagged with ITHE True), unless it's
        # already included
        if not calculated_nav['ITHEs'].iloc[-1]:  # Check if the last value of ITHE is False
            excess_losses_at_ithes = pd.concat(
                [excess_losses_at_ithes, pd.Series([last_excess_loss], index=[calculated_nav.index[-1]])])
        if excess_losses_at_ithes.empty:  # Check if excess_losses_at_ithes is empty
            el_cv = np.nan  # Return NaN or some other appropriate value
        else:
            el_cv = np.std(excess_losses_at_ithes) / np.mean(excess_losses_at_ithes)

        # penalty_constant = 0.2  # This value can be adjusted
        # penalty = penalty_constant / np.log(len(ith_dates)+1)
        # aggcv = (np.mean(np.array([el_cv, ith_cv])**2) / (num_of_ith_epochs**2) * 100) + penalty
        # aggcv = (np.mean(np.array([el_cv, ith_cv])**2) / (num_of_ith_epochs**2) * 100)
        aggcv = max(el_cv, ith_cv)
        print(f'{aggcv=}')

        if aggcv_low_bound < aggcv < aggcv_up_bound:
            # filename = f"{aggcv:.5f}_cv_{el_cv:.4f}_ELcv_{ith_cv:.4f}_ITHcv_{cumulative_sum:.4f}_Tn_{
            # calculated_nav['NAV'].max():.4f}_Mn_{len(ith_dates) - 2}_ITHEs_{sharpe_ratio:.4f}_sr.html"  # Subtract
            # 2 from len(ith_dates) to exclude start and end dates
            filename = f"EL_{el_cv:.5f}_ITHC_{ith_cv:.5f}_TMAEG_{TMAEG:.5f}_ITHEs_{ithe_ct}_D2ITHE_{days_taken_to_ithe:.2f}_SR_{sharpe_ratio:.4f}_UID_{uid}.html"  # Subtract 2 from len(ith_dates) to exclude start and end dates

            # Create and save plot
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01,
                                subplot_titles=('NAV', 'Excess Gains & Losses'))

            ith_epochs = calculated_nav[calculated_nav['ITHEs']].index  # Extract the ith epochs
            num_of_ith_epochs = len(ith_epochs)  # Count the number of ith epochs
            ithes_dir = output_dir / f"ITHEs_{num_of_ith_epochs}"  # Create a subdirectory for this number of ITHEs
            ithes_dir.mkdir(parents=True, exist_ok=True)
            crossover_epochs = calculated_nav.loc[ith_epochs]
            fig.add_trace(
                go.Scatter(x=crossover_epochs.index, y=crossover_epochs['NAV'], mode='markers', name='ITHEs on NAV',
                           marker=dict(color='darkgoldenrod', size=20)), row=1, col=1)
            fig.add_trace(go.Scatter(x=crossover_epochs.index, y=crossover_epochs['Excess Gains'], mode='markers',
                                     name='ITHEs on Excess Gains', marker=dict(color='blue', size=20)), row=2, col=1)
            fig.add_trace(go.Scatter(x=calculated_nav.index, y=calculated_nav['NAV'], mode='lines', name='NAV'), row=1,
                          col=1)
            fig.add_trace(
                go.Scatter(x=calculated_nav.index, y=calculated_nav['Excess Gains'], mode='lines', name='Excess Gains',
                           line=dict(color='green')), row=2, col=1)
            fig.add_trace(go.Scatter(x=calculated_nav.index, y=calculated_nav['Excess Losses'], mode='lines',
                                     name='Excess Losses', line=dict(color='red')), row=2, col=1)
            fig.update_layout(
                title=f'{num_of_ith_epochs} ITH Epochs -- {filename}',
                autosize=False,
                width=2040,  # entire canvas width
                height=1200,
                margin=dict(l=50, r=50, b=50, t=50, pad=4),
                paper_bgcolor="DarkSlateGrey",
                plot_bgcolor='Black',
                legend=dict(
                    x=0.01,
                    y=0.98,
                    traceorder="normal",
                    font=dict(
                        family="Courier New",  # This line changes the font to Monospace
                        size=12,
                        color="White"
                    ),
                    bgcolor="DarkSlateGrey",
                    bordercolor="White",
                    borderwidth=2
                ),
                font=dict(  # This line changes the global font to Monospace
                    family="Monospace",
                    size=12,
                    color="White"
                ),
                annotations=[
                    dict(
                        x=0.5,  # x position (0:left, 1:right)
                        y=0.95,  # y position (0:bottom, 1:top)
                        xref='paper',  # x position refers to the whole area (including margins)
                        yref='paper',  # y position refers to the whole area (including margins)
                        text='NAV<br>',
                        showarrow=False,
                        font=dict(
                            size=16,
                            color="White"
                        ),
                    ),
                    dict(
                        x=0.5,  # x position (0:left, 1:right)
                        y=0.45,  # y position (0:bottom, 1:top)
                        xref='paper',  # x position refers to the whole area (including margins)
                        yref='paper',  # y position refers to the whole area (including margins)
                        text='Excess Gains & Losses<br>',
                        showarrow=False,
                        font=dict(
                            size=16,
                            color="White"
                        ),
                    ),
                ]
            )
            fig.update_yaxes(gridcolor='dimgray', type="linear", row=1,
                             col=1)  # This line sets the y-axis of the first subplot to a log scale
            fig.update_yaxes(gridcolor='dimgray', row=2, col=1)
            fig.update_xaxes(gridcolor='dimgray', row=1, col=1)
            fig.update_xaxes(gridcolor='dimgray', row=2, col=1)

            # Generate monthly ticks between the minimum and maximum dates
            monthly_ticks = pd.date_range(nav_data.index.min(), nav_data.index.max(), freq='MS')
            monthly_tick_labels = monthly_ticks.strftime('%Y-%m')

            # Customize X-axis grid lines
            custom_xaxis = XAxis(
                tickmode='array',
                tickvals=monthly_ticks,  # Set to monthly_ticks
                showgrid=True,  # Show vertical grid
                gridwidth=0.5  # Vertical grid width
            )

            custom_yaxis = YAxis(
                showgrid=True,  # Show vertical grid
                gridwidth=0.5  # Vertical grid width
            )
            fig.update_layout(xaxis=custom_xaxis.to_plotly_json())
            fig.update_layout(yaxis=custom_yaxis.to_plotly_json())

            fig.show()

            # In your while loop, after generating the HTML file:
            filename_with_path = str(ithes_dir / filename)
            filename_with_path_main_output = str(output_dir / filename)  # Save also to main output directory

            fig.layout.template = 'plotly_dark'
            fig.write_html(filename_with_path)
            fig.write_html(filename_with_path_main_output)

            # Add the uid as a comment to the file
            add_comment_to_file_finder_column(filename_with_path, uid)
            add_comment_to_file_finder_column(filename_with_path_main_output, uid)

            # Save synthetic NAV data to CSV file only if it meets the criteria
            csv_filename = filename.replace('.html', '.csv')
            csv_filename_with_path = str(nav_dir / csv_filename)
            nav_data.to_csv(csv_filename_with_path)

            add_comment_to_file_finder_column(csv_filename_with_path, uid)

            # Increment qualified_results
            qualified_results += 1

    if ith_durations is not None:
        print(f"ith_durations in process_nav_data: {ith_durations}")

    print(f"excess_losses_at_ithes in process_nav_data: {excess_losses_at_ithes}")

    # return these variables at the end
    return qualified_results, sharpe_ratio, num_of_ith_epochs, filename, uid


# Set constants
TMAEG_dynamically_determined_by = "geomean"
TMAEG_dynamically_determined_by = "fixed"  # TMAEG can set to a fixed but a higher value than that of the HMFD
TMAEG_dynamically_determined_by = "mdd"  # TMAEG is the historical maximum fractional drawdown (HMFD) from a all-time
# peak by default
TMAEG = 0.05
date_initiate = '2022-06-15'
date_conclude = '2023-11-16'
date_duration = (pd.to_datetime(date_conclude) - pd.to_datetime(date_initiate)).days
ith_epochs_lower_bound = int(np.floor(date_duration / 28 / 6))
print(f'{ith_epochs_lower_bound=}')
ith_epochs_lower_bound = 0
ith_epochs_upper_bound = 100000
sr_lower_bound = 1.00
sr_upper_bound = 100000.00
aggcv_low_bound = 0
aggcv_up_bound = 1110.00
# Initialize counters
qualified_results = 0

# Try to load existing data
existing_csv_files = glob.glob(str(nav_dir / '*.csv'))

print(f'{existing_csv_files=}')

for i, csv_file in enumerate(existing_csv_files, 1):
    print(csv_file)
    nav_data = pd.read_csv(csv_file, index_col='Date', parse_dates=True)

    # Check if PnL column is missing and calculate it if needed
    if 'PnL' not in nav_data.columns:
        print(f"PnL column missing in {csv_file}. Calculating from NAV...")
        nav_data = calculate_pnl_from_nav(nav_data)

    if TMAEG_dynamically_determined_by == "geomean":
        TMAEG = calculate_geometric_mean_of_drawdown(nav_data['NAV'])
    elif TMAEG_dynamically_determined_by == "mdd":
        TMAEG = calculate_max_drawdown(nav_data['NAV'])
    elif TMAEG_dynamically_determined_by == "fixed":
        TMAEG
    qualified_results, sharpe_ratio, num_of_ith_epochs, filename, uid = process_nav_data(nav_data, output_dir,
                                                                                         qualified_results, nav_dir,
                                                                                         TMAEG, instrument=False)
    print(
        f'Processing file {i:4} of {len(existing_csv_files)}: {filename=}, {uid=}, {TMAEG=}, {sharpe_ratio=}, from '
        f'CSV generated.',
        end='\n')

# Generate new NAV data if necessary
counter = 0
while qualified_results < 5:
    counter += 1
    synthetic_nav = generate_synthetic_nav(date_initiate, date_conclude, avg_daily_return=0.0010123,
                                           daily_return_volatility=0.09, df=10)
    if TMAEG_dynamically_determined_by == "geomean":
        TMAEG = calculate_geometric_mean_of_drawdown(synthetic_nav['NAV'])
    elif TMAEG_dynamically_determined_by == "mdd":
        TMAEG = calculate_max_drawdown(synthetic_nav['NAV'])
    elif TMAEG_dynamically_determined_by == "fixed":
        TMAEG
    qualified_results, sharpe_ratio, num_of_ith_epochs, filename, uid = process_nav_data(synthetic_nav, output_dir,
                                                                                         qualified_results, nav_dir,
                                                                                         TMAEG)

    if filename is not None:
        print(
            f'Processing synthetic data {counter:4}: {filename=}, {uid=}, {TMAEG=}, {sharpe_ratio=}, newly generated.')
        # Save synthetic NAV data to CSV file
        csv_filename = filename.replace('.html', '.csv')
        csv_filename_with_path = str(nav_dir / csv_filename)
        synthetic_nav.to_csv(csv_filename_with_path)
