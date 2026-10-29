import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import data
import portfolio


def load_nse_companies():
    """Load NSE companies from CSV."""
    df = pd.read_csv('nse_companies.csv')
    return df


def main():
    st.set_page_config(page_title="Portfolio Optimiser", layout="wide")
    
    st.title("Portfolio Optimiser")
    st.markdown("*Educational use only. Not investment advice. Past returns do not guarantee future returns.*")
    
    # Load NSE companies
    nse_companies = load_nse_companies()
    
    # Sidebar inputs
    st.sidebar.header("Inputs")
    
    # 1.1 Stocks multiselect
    stock_options = nse_companies['company_name'].tolist()
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks (2-25)",
        options=stock_options,
        default=["HDFC Bank", "TCS", "Sun Pharma", "ITC", "L&T"],
        key="stock_select"
    )
    
    # 1.2 Tenure
    tenure_options = ["1 year", "3 years", "5 years", "Custom"]
    tenure = st.sidebar.radio("Tenure", options=tenure_options, index=0, key="tenure")
    
    end_date = datetime.now().date()
    
    if tenure == "1 year":
        start_date = end_date - timedelta(days=365)
    elif tenure == "3 years":
        start_date = end_date - timedelta(days=365*3)
    elif tenure == "5 years":
        start_date = end_date - timedelta(days=365*5)
    else:  # Custom
        start_date = st.sidebar.date_input("Start Date", value=end_date - timedelta(days=365), key="start_date")
        end_date = st.sidebar.date_input("End Date", value=end_date, key="end_date")
    
    # 1.3 Risk-free rate
    risk_free_rate = st.sidebar.number_input(
        "Risk-free rate (% per year)",
        value=6.5,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        key="risk_free_rate"
    )
    
    # 1.4 Max weight per stock
    max_weight_percent = st.sidebar.slider(
        "Max weight per stock (%)",
        min_value=10,
        max_value=100,
        value=100,
        key="max_weight"
    )
    max_weight = max_weight_percent / 100
    
    # Calculate button
    calculate = st.sidebar.button("Calculate", key="calculate")
    
    # Validation
    if len(selected_stocks) < 2:
        st.error("Please select at least 2 stocks.")
        return
    if len(selected_stocks) > 25:
        st.error("Please select at most 25 stocks.")
        return
    
    min_required_weight = 1 / len(selected_stocks)
    if max_weight < min_required_weight:
        st.error(f"Max weight must be at least {min_required_weight:.2%} ({min_required_weight*100:.2f}%) for {len(selected_stocks)} stocks.")
        return
    
    # 1.5 Manual weights
    st.sidebar.subheader("Manual Weights (%)")
    manual_weights = {}
    total_manual = 0.0
    for stock in selected_stocks:
        default_weight = 100 / len(selected_stocks)
        weight = st.sidebar.number_input(
            f"{stock}",
            value=default_weight,
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            key=f"manual_{stock}"
        )
        manual_weights[stock] = weight
        total_manual += weight
    
    st.sidebar.write(f"Total: {total_manual:.2f}%")
    
    if not calculate:
        st.info("Click 'Calculate' in the sidebar to run the analysis.")
        return
    
    # Data download
    try:
        with st.spinner("Downloading data..."):
            prices = data.prepare_data(selected_stocks, start_date, end_date)
    except ValueError as e:
        st.error(str(e))
        return
    
    # Check row count warning
    expected_rows = {
        "1 year": 245,
        "3 years": 735,
        "5 years": 1225
    }
    if tenure in expected_rows and len(prices) < expected_rows[tenure]:
        st.warning(f"Expected approximately {expected_rows[tenure]} rows for {tenure}, but got {len(prices)} rows.")
    
    # Calculate returns
    returns = data.calculate_returns(prices)
    
    # Averages
    averages = portfolio.calculate_averages(returns)
    
    # Covariance
    cov_matrix = portfolio.calculate_covariance(returns)
    
    # Stock names for display
    stock_names = selected_stocks
    
    # Tab 1: Results
    tab1, tab2, tab3 = st.tabs(["Results", "Efficient Frontier", "How it Works"])
    
    with tab1:
        # 8.1 Prices and returns table
        st.subheader("Prices and Returns")
        display_data = pd.DataFrame()
        display_data['Date'] = prices.index
        for stock in stock_names:
            display_data[f'{stock} Close'] = prices[stock].values
        for stock in stock_names:
            # Add returns column (first row has no return)
            return_col = [np.nan] + returns[stock].tolist()
            display_data[f'{stock} Return'] = return_col
        st.dataframe(display_data, use_container_width=True)
        
        # 8.2 Averages table
        st.subheader("Average Daily Returns")
        averages_df = pd.DataFrame({
            'Stock': stock_names,
            'Average (decimal)': averages.values,
            'Average (%)': (averages.values * 100)
        })
        st.dataframe(averages_df, use_container_width=True)
        
        # 8.3 Covariance table
        st.subheader("Covariance Matrix (Population)")
        cov_df = pd.DataFrame(cov_matrix, index=stock_names, columns=stock_names)
        st.dataframe(cov_df, use_container_width=True)
        
        # Manual portfolio (only if weights total 100%)
        manual_weights_valid = abs(total_manual - 100) < 0.01
        if manual_weights_valid:
            st.subheader("Manual Portfolio Results")
            manual_w = np.array([manual_weights[stock] / 100 for stock in stock_names])
            manual_expected = portfolio.calculate_expected_return(manual_w, averages)
            first_term, second_term, final_second_term, variance, risk = portfolio.calculate_portfolio_risk(manual_w, cov_matrix)
            return_per_risk, rf_daily = portfolio.calculate_sharpe_ratio(manual_expected, risk, risk_free_rate)
            annual_return, annual_risk, annual_ratio = portfolio.annualise_metrics(manual_expected, risk, risk_free_rate)
            
            manual_results = {
                'Stock': stock_names,
                'Weight (%)': [manual_weights[stock] for stock in stock_names]
            }
            st.dataframe(pd.DataFrame(manual_results), use_container_width=True)
            
            st.write(f"Expected Return (daily): {manual_expected:.6f} ({manual_expected*100:.6f}%)")
            st.write(f"First Term: {first_term:.10f}")
            st.write(f"Second Term: {second_term:.10f}")
            st.write(f"Final Second Term: {final_second_term:.10f}")
            st.write(f"Variance: {variance:.10f}")
            st.write(f"Risk (daily): {risk:.6f} ({risk*100:.6f}%)")
            st.write(f"Return/Risk: {return_per_risk if isinstance(return_per_risk, str) else return_per_risk:.6f}")
            st.write(f"Risk-free rate (daily): {rf_daily:.10f}")
            st.write(f"Annual Return: {annual_return:.6f} ({annual_return*100:.6f}%)")
            st.write(f"Annual Risk: {annual_risk:.6f} ({annual_risk*100:.6f}%)")
            st.write(f"Annual Ratio: {annual_ratio if isinstance(annual_ratio, str) else annual_ratio:.6f}")
        else:
            st.error(f"Manual weights total {total_manual:.2f}%, not 100%. Manual portfolio not calculated.")
        
        # 8.5 Optimised portfolio
        st.subheader("Optimised Portfolio Results")
        try:
            opt_weights = portfolio.optimize_portfolio(averages, cov_matrix, max_weight)
            opt_expected = portfolio.calculate_expected_return(opt_weights, averages)
            _, _, _, _, opt_risk = portfolio.calculate_portfolio_risk(opt_weights, cov_matrix)
            opt_return_per_risk, _ = portfolio.calculate_sharpe_ratio(opt_expected, opt_risk, risk_free_rate)
            opt_annual_return, opt_annual_risk, opt_annual_ratio = portfolio.annualise_metrics(opt_expected, opt_risk, risk_free_rate)
            
            # Weights table
            opt_weights_df = pd.DataFrame({
                'Stock': stock_names,
                'Weight (%)': (opt_weights * 100).round(2)
            })
            st.dataframe(opt_weights_df, use_container_width=True)
            
            # Pie chart (hide 0% stocks)
            non_zero = opt_weights > 0
            if np.any(non_zero):
                pie_df = pd.DataFrame({
                    'Stock': [stock_names[i] for i in range(len(stock_names)) if non_zero[i]],
                    'Weight (%)': [(opt_weights[i] * 100).round(2) for i in range(len(stock_names)) if non_zero[i]]
                })
                fig_pie = px.pie(pie_df, values='Weight (%)', names='Stock', title='Optimised Portfolio Weights')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.write(f"Expected Return (daily): {opt_expected:.6f} ({opt_expected*100:.6f}%)")
            st.write(f"Risk (daily): {opt_risk:.6f} ({opt_risk*100:.6f}%)")
            st.write(f"Return/Risk: {opt_return_per_risk if isinstance(opt_return_per_risk, str) else opt_return_per_risk:.6f}")
            st.write(f"Annual Return: {opt_annual_return:.6f} ({opt_annual_return*100:.6f}%)")
            st.write(f"Annual Risk: {opt_annual_risk:.6f} ({opt_annual_risk*100:.6f}%)")
            st.write(f"Annual Ratio: {opt_annual_ratio if isinstance(opt_annual_ratio, str) else opt_annual_ratio:.6f}")
        except ValueError as e:
            st.error(str(e))
        
        # 8.6 Comparison table
        st.subheader("Comparison: Manual vs Optimised vs Equal-weight")
        equal_weights = np.ones(len(stock_names)) / len(stock_names)
        equal_expected = portfolio.calculate_expected_return(equal_weights, averages)
        _, _, _, _, equal_risk = portfolio.calculate_portfolio_risk(equal_weights, cov_matrix)
        equal_return_per_risk, _ = portfolio.calculate_sharpe_ratio(equal_expected, equal_risk, risk_free_rate)
        equal_annual_return, equal_annual_risk, equal_annual_ratio = portfolio.annualise_metrics(equal_expected, equal_risk, risk_free_rate)
        
        comparison_data = []
        if manual_weights_valid:
            comparison_data.append({
                'Portfolio': 'Manual',
                'Expected Return (daily)': f"{manual_expected:.6f}",
                'Risk (daily)': f"{risk:.6f}",
                'Return/Risk': f"{return_per_risk if isinstance(return_per_risk, str) else return_per_risk:.6f}",
                'Annual Return': f"{annual_return:.6f}",
                'Annual Risk': f"{annual_risk:.6f}",
                'Annual Ratio': f"{annual_ratio if isinstance(annual_ratio, str) else annual_ratio:.6f}"
            })
        
        comparison_data.append({
            'Portfolio': 'Optimised',
            'Expected Return (daily)': f"{opt_expected:.6f}",
            'Risk (daily)': f"{opt_risk:.6f}",
            'Return/Risk': f"{opt_return_per_risk if isinstance(opt_return_per_risk, str) else opt_return_per_risk:.6f}",
            'Annual Return': f"{opt_annual_return:.6f}",
            'Annual Risk': f"{opt_annual_risk:.6f}",
            'Annual Ratio': f"{opt_annual_ratio if isinstance(opt_annual_ratio, str) else opt_annual_ratio:.6f}"
        })
        
        comparison_data.append({
            'Portfolio': 'Equal-weight',
            'Expected Return (daily)': f"{equal_expected:.6f}",
            'Risk (daily)': f"{equal_risk:.6f}",
            'Return/Risk': f"{equal_return_per_risk if isinstance(equal_return_per_risk, str) else equal_return_per_risk:.6f}",
            'Annual Return': f"{equal_annual_return:.6f}",
            'Annual Risk': f"{equal_annual_risk:.6f}",
            'Annual Ratio': f"{equal_annual_ratio if isinstance(equal_annual_ratio, str) else equal_annual_ratio:.6f}"
        })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # 8.8 Download button
        st.subheader("Download Results")
        
        # Create Excel file with all tables
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            display_data.to_excel(writer, sheet_name='Prices and Returns', index=False)
            averages_df.to_excel(writer, sheet_name='Averages', index=False)
            cov_df.to_excel(writer, sheet_name='Covariance', index=True)
            pd.DataFrame(comparison_data).to_excel(writer, sheet_name='Comparison', index=False)
            opt_weights_df.to_excel(writer, sheet_name='Optimised Weights', index=False)
        
        buffer.seek(0)
        st.download_button(
            label="Download Excel",
            data=buffer,
            file_name="portfolio_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with tab2:
        # 8.7 Correlation heatmap and efficient frontier
        st.subheader("Correlation Heatmap")
        corr_matrix = np.corrcoef(returns.T)
        corr_df = pd.DataFrame(corr_matrix, index=stock_names, columns=stock_names)
        fig_heatmap = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.subheader("Efficient Frontier")
        # Generate random long-only portfolios
        np.random.seed(42)
        n_portfolios = 1000
        random_portfolios = []
        for _ in range(n_portfolios):
            w = dirichlet.rvs(np.ones(len(stock_names)), size=1)[0]
            # Apply max weight constraint
            w = np.minimum(w, max_weight)
            w = w / np.sum(w)
            exp_ret = portfolio.calculate_expected_return(w, averages)
            _, _, _, _, port_risk = portfolio.calculate_portfolio_risk(w, cov_matrix)
            sharpe = exp_ret / port_risk if port_risk > 0 else 0
            random_portfolios.append({'Risk': port_risk, 'Return': exp_ret, 'Sharpe': sharpe})
        
        random_df = pd.DataFrame(random_portfolios)
        
        fig_frontier = go.Figure()
        fig_frontier.add_trace(go.Scatter(
            x=random_df['Risk'],
            y=random_df['Return'],
            mode='markers',
            marker=dict(
                color=random_df['Sharpe'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Return/Risk')
            ),
            name='Random Portfolios'
        ))
        
        # Add marked portfolios
        if manual_weights_valid:
            fig_frontier.add_trace(go.Scatter(
                x=[risk],
                y=[manual_expected],
                mode='markers',
                marker=dict(size=15, color='red', symbol='circle'),
                name='Manual'
            ))
        
        fig_frontier.add_trace(go.Scatter(
            x=[opt_risk],
            y=[opt_expected],
            mode='markers',
            marker=dict(size=15, color='green', symbol='star'),
            name='Optimised'
        ))
        
        fig_frontier.add_trace(go.Scatter(
            x=[equal_risk],
            y=[equal_expected],
            mode='markers',
            marker=dict(size=15, color='blue', symbol='diamond'),
            name='Equal-weight'
        ))
        
        fig_frontier.update_layout(
            xaxis_title='Risk (daily)',
            yaxis_title='Expected Return (daily)',
            title='Efficient Frontier'
        )
        st.plotly_chart(fig_frontier, use_container_width=True)
    
    with tab3:
        # 8.9 How it works
        st.subheader("How it Works")
        
        st.markdown("""
        ### 3. Returns Calculation
        Daily returns are calculated using simple percentage change:
        ```
        daily_return(t) = (close(t) - close(t-1)) / close(t-1)
        ```
        The first date has no return and is dropped. Returns are stored as decimals (0.02 means 2%).
        
        ### 4. Averages and Expected Return
        - **Average**: Arithmetic mean of each stock's daily returns (like Excel AVERAGE)
        - **Expected Return**: Weighted sum of averages using portfolio weights (like Excel SUMPRODUCT)
        ```
        expected_return(w) = sum over i of (w_i * average_i)
        ```
        
        ### 5. Covariance Table
        Population covariance matrix (N x N) using the formula dividing by n (not n-1):
        ```
        cov_ij = sum over t of ((r_i(t) - avg_i) * (r_j(t) - avg_j)) / n
        ```
        The diagonal contains population variances (like Excel VARP).
        
        ### 6. Portfolio Risk
        Calculated in this exact sequence:
        1. **First term**: sum of (weight_i² × variance_i)
        2. **Second term**: sum of (weight_i × weight_j × covariance_ij) for all pairs i < j
        3. **Final second term**: 2 × second term
        4. **Variance**: first term + final second term
        5. **Risk**: sqrt(variance)
        6. **Return per risk**: (expected_return - risk_free_daily) / risk
        
        ### 7. Optimisation
        The optimiser maximises the return_per_risk ratio (Sharpe-like ratio) using:
        - **Objective**: Maximise return_per_risk (minimise its negative)
        - **Constraints**: Weights sum to 1, each weight between 0 and max_weight
        - **Method**: scipy.optimize.minimize with SLSQP algorithm
        - **Starting points**: 50 random Dirichlet distributions + equal-weight start
        - The best solution among successful runs is selected
        """)
        
        st.warning("Educational use only. Not investment advice. Past returns do not guarantee future returns.")


if __name__ == "__main__":
    main()
