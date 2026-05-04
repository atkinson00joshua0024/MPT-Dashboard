import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import requests
import zipfile
import io as _io


def run_mpt_simulation(tickers, start_date, end_date, num_simulations, risk_free_rate):
    """
    Performs a Markowitz Portfolio Optimization using Monte Carlo simulation.
    """
    with st.spinner(f"Downloading data for {len(tickers)} stocks..."):
        adj_close_df = pd.DataFrame()
        for ticker in tickers:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            adj_close_df[ticker] = data['Close']

    returns = adj_close_df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    with st.spinner(f"Running {num_simulations} simulations..."):
        results = np.zeros((3 + len(tickers), num_simulations))

        for i in range(num_simulations):
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)

            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = sharpe_ratio
            for j in range(len(weights)):
                results[j + 3, i] = weights[j]

    results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe'] + tickers)
    return results_df, returns


# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title('Portfolio Optimization & Analysis Dashboard')

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header('Configuration')

    tickers_string = st.text_area(
        'Stock Tickers (comma-separated)',
        'MSFT, AAPL, UNH, JNJ, JPM, V, AMZN, TSLA, GOOGL, META, CAT, RTX, PG, WMT, XOM, CVX, NEE, DUK'
    )

    start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2025-07-26'))

    num_simulations = st.slider('Number of Simulations', 1000, 50000, 10000)
    risk_free_rate = st.slider('Risk-Free Rate (%)', 0.0, 5.0, 2.0) / 100

    run_button = st.button('Run Full Analysis')

# --- Main Panel for Outputs ---
if run_button:
    tickers = [ticker.strip().upper() for ticker in tickers_string.split(',')]

    results_df, returns = run_mpt_simulation(tickers, start_date, end_date, num_simulations, risk_free_rate)

    max_sharpe_portfolio = results_df.iloc[results_df['Sharpe'].idxmax()]
    min_volatility_portfolio = results_df.iloc[results_df['Volatility'].idxmin()]

    # --- Create Tabs for Each Section ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Optimal Portfolios",
        "🔥 Correlation Heatmap",
        "📊 Factor Analysis",
        "📉 Stress Testing",
        "🌐 Efficient Frontier",
        "💡 Explanations"
    ])

    with tab1:
        st.header('Optimal Portfolio Allocations')

        col1, col2 = st.columns(2)

        allocation_sharpe = max_sharpe_portfolio.drop(['Return', 'Volatility', 'Sharpe'])
        allocation_min_vol = min_volatility_portfolio.drop(['Return', 'Volatility', 'Sharpe'])

        combined_index = allocation_sharpe[allocation_sharpe > 0.001].index.union(
            allocation_min_vol[allocation_min_vol > 0.001].index
        )

        display_sharpe = allocation_sharpe.reindex(combined_index, fill_value=0).sort_values(ascending=False)
        display_min_vol = allocation_min_vol.reindex(display_sharpe.index)

        with col1:
            st.subheader('Max Sharpe Ratio Portfolio')
            st.write(f"**Annual Return:** {max_sharpe_portfolio['Return'] * 100:.2f}%")
            st.write(f"**Annual Volatility:** {max_sharpe_portfolio['Volatility'] * 100:.2f}%")
            st.write(f"**Sharpe Ratio:** {max_sharpe_portfolio['Sharpe']:.2f}")

            st.dataframe(display_sharpe.apply(lambda x: f"{x * 100:.2f}%"), column_config={"value": "Allocation"})

        with col2:
            st.subheader('Minimum Volatility Portfolio')
            st.write(f"**Annual Return:** {min_volatility_portfolio['Return'] * 100:.2f}%")
            st.write(f"**Annual Volatility:** {min_volatility_portfolio['Volatility'] * 100:.2f}%")
            st.write(f"**Sharpe Ratio:** {min_volatility_portfolio['Sharpe']:.2f}")

            st.dataframe(display_min_vol.apply(lambda x: f"{x * 100:.2f}%"), column_config={"value": "Allocation"})

    with tab2:
        st.header('Optimized Portfolio Correlation Heatmap')
        st.write(
            "This heatmap shows how closely the stocks in your Max Sharpe Ratio portfolio move together. Lower numbers suggest better diversification.")

        significant_weights = allocation_sharpe[allocation_sharpe > 0.01]
        significant_tickers = significant_weights.index.tolist()

        corr_matrix = returns[significant_tickers].corr()

        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        # --- THIS LINE IS NOW FIXED ---
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

    with tab3:
        st.header('Fama-French 3-Factor Model Exposure')
        st.write(
            "This analysis shows how much of your portfolio's performance can be explained by common market risk factors (Market, Size, Value).")

        with st.spinner("Downloading Fama-French data..."):
            ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
            r = requests.get(ff_url, timeout=30)
            z = zipfile.ZipFile(_io.BytesIO(r.content))
            csv_name = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
            with z.open(csv_name) as f:
                raw = f.read().decode('utf-8')
            lines = raw.split('\n')
            header_idx = next(i for i, l in enumerate(lines) if 'Mkt-RF' in l)
            end_idx = next((i for i, l in enumerate(lines) if i > header_idx + 1 and l.strip() == ''), len(lines))
            ff_csv = '\n'.join(lines[header_idx:end_idx])
            ff_factors = pd.read_csv(_io.StringIO(ff_csv), index_col=0)
            ff_factors.index = pd.to_datetime(ff_factors.index.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
            ff_factors.columns = [c.strip() for c in ff_factors.columns]
            ff_factors = ff_factors.apply(pd.to_numeric, errors='coerce').dropna()
            ff_factors = ff_factors / 100
            ff_factors = ff_factors[ff_factors.index >= pd.to_datetime(start_date)]

            portfolio_weights = max_sharpe_portfolio.drop(['Return', 'Volatility', 'Sharpe'])
            portfolio_returns = (returns * portfolio_weights).sum(axis=1)

            merged_data = pd.merge(portfolio_returns.rename('Portfolio'), ff_factors, left_index=True, right_index=True)
            merged_data['Portfolio_Excess'] = merged_data['Portfolio'] - merged_data['RF']

            X = merged_data[['Mkt-RF', 'SMB', 'HML']]
            y = merged_data['Portfolio_Excess']
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            st.subheader('Regression Results')

            alpha_annual = model.params['const'] * 252
            r_squared = model.rsquared

            st.metric(label="Annual Alpha (Jensen's Alpha)", value=f"{alpha_annual * 100:.4f}%")
            st.metric(label="R-squared", value=f"{r_squared:.3f}")

            results_summary = pd.DataFrame({
                "Coefficient": model.params,
                "Standard Error": model.bse,
                "P-value": model.pvalues
            })

            st.write("Factor Exposures:")
            st.dataframe(results_summary)

    with tab4:
        st.header('Portfolio Stress Testing')
        st.write(
            "This section simulates how your optimized portfolio would have performed during historical market crises.")

        stress_scenarios = {
            "COVID-19 Crash": ('2020-02-19', '2020-03-23'),
            "2008 Financial Crisis": ('2008-09-15', '2009-03-09')
        }

        for name, (start, end) in stress_scenarios.items():
            st.subheader(f"Scenario: {name} ({start} to {end})")

            try:
                stress_data = yf.download(tickers, start=start, end=end, progress=False)['Close']
                stress_returns = stress_data.pct_change().dropna()

                portfolio_weights = max_sharpe_portfolio.drop(['Return', 'Volatility', 'Sharpe'])
                stress_portfolio_returns = (stress_returns * portfolio_weights).sum(axis=1)
                cumulative_returns = (1 + stress_portfolio_returns).cumprod() - 1

                total_return = cumulative_returns.iloc[-1]
                max_drawdown = (cumulative_returns / (cumulative_returns.cummax() + 1) - 1).min()

                st.write(f"**Total Return:** {total_return * 100:.2f}%")
                st.write(f"**Maximum Drawdown:** {max_drawdown * 100:.2f}%")

                fig_stress, ax_stress = plt.subplots(figsize=(10, 5))
                cumulative_returns.plot(ax=ax_stress, title=f"Portfolio Performance during {name}")
                ax_stress.set_ylabel("Cumulative Return")
                st.pyplot(fig_stress)

            except Exception as e:
                st.warning(
                    f"Could not run '{name}' scenario. Data for this period may not be available for all selected stocks.")

    with tab5:
        st.header('Efficient Frontier')

        fig_ef, ax_ef = plt.subplots(figsize=(12, 7))
        scatter = ax_ef.scatter(results_df.Volatility, results_df.Return, c=results_df.Sharpe, cmap='viridis')
        plt.colorbar(scatter, label='Sharpe Ratio')

        ax_ef.scatter(max_sharpe_portfolio.Volatility, max_sharpe_portfolio.Return, marker='*', color='r', s=500,
                      label='Max Sharpe Ratio')
        ax_ef.scatter(min_volatility_portfolio.Volatility, min_volatility_portfolio.Return, marker='*', color='orange',
                      s=500, label='Min Volatility')

        ax_ef.set_title('Monte Carlo Simulation of Portfolios')
        ax_ef.set_xlabel('Annualized Volatility (Risk)')
        ax_ef.set_ylabel('Annualized Return')
        ax_ef.legend(labelspacing=0.8)

        st.pyplot(fig_ef)

    with tab6:
        st.header("Explanations of Key Terms")

        st.subheader("Optimal Portfolios")
        st.markdown("""
        - **Annual Return**: The average return you can expect from the portfolio over a year.
        - **Annual Volatility (Risk)**: A measure of how much the portfolio's value is likely to fluctuate. A higher number means higher risk.
        - **Sharpe Ratio**: A measure of risk-adjusted return. It tells you how much return you get for each unit of risk you take. A higher Sharpe Ratio is better.
        - **Max Sharpe Ratio Portfolio**: The single best portfolio found by the simulation that provides the highest return for the amount of risk taken.
        - **Minimum Volatility Portfolio**: The portfolio with the absolute lowest risk, regardless of its return.
        """)

        st.subheader("Correlation Heatmap")
        st.markdown("""
        - **Correlation**: A measure of how two stocks move in relation to each other.
        - **How to Read the Heatmap**: The values range from -1 to 1.
            - **+1 (Hot Color)**: The stocks move in perfect lockstep.
            - **0 (Neutral Color)**: There is no relationship between their movements.
            - **-1 (Cool Color)**: The stocks move in opposite directions.
        - **Why it matters**: A well-diversified portfolio should ideally have stocks with low correlation to each other.
        """)

        st.subheader("Factor Analysis")
        st.markdown("""
        - **Alpha**: The portion of the portfolio's return that is *not* explained by the market factors. A positive alpha can suggest that the portfolio's strategy added value.
        - **R-squared**: A value between 0 and 1 that tells you how much of your portfolio's performance is explained by the model's factors. A high R-squared (e.g., 0.95) means the portfolio closely tracks the market.
        - **Coefficients (coef)**: These numbers show the portfolio's exposure to each factor. A `Mkt-RF` coefficient of 1.0 means the portfolio moves in line with the market.
        """)

        st.subheader("Stress Testing")
        st.markdown("""
        - **Total Return**: The total percentage gain or loss the portfolio would have experienced during the historical crisis period.
        - **Maximum Drawdown**: The largest single drop from a peak to a trough during the period. It's a key measure of how much an investor might lose in a worst-case scenario.
        """)
