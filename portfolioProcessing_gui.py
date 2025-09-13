import pandas as pd
import numpy as np
import os
import sys
import threading
from tkinter import Tk, filedialog, simpledialog, messagebox, Button, Label, Toplevel, Frame, Entry, StringVar
from tkinter.constants import LEFT, RIGHT, TOP, BOTTOM, X, Y, BOTH
from tkinter.scrolledtext import ScrolledText

from portfolio_helpers import append_holding_info

# Map all possible column names to standardized names
COLUMN_MAP = {
    'Symbol': 'Symbol',
    'Ticker': 'Symbol',
    'Description': 'Description',
    'Security Name': 'Description',
    'Qty (Quantity)': 'Quantity',
    'Quantity': 'Quantity',
    'Last Price': 'Price',
    'Price': 'Price',
    'Mkt Val (Market Value)': 'Market Value',
    'Current Value': 'Market Value',
    'Market Value': 'Market Value',
    'Cost Basis': 'Cost Basis',
    'Cost Basis Total': 'Cost Basis',
    'Gain $ (Gain/Loss $)': 'Gain/Loss $',
    'Total Gain/Loss Dollar': 'Gain/Loss $',
    'Security Type': 'Type',
    'Type': 'Type',
    '% of Acct (% of Account)': '% of Account',
    'Percent Of Account': '% of Account'
}

KEY_COLS = [
    'Portfolio Name',
    'Symbol',
    'Description',
    'Quantity',
    'Price',
    'Market Value',
    'Cost Basis',
    'Gain/Loss $',
    'Type',
    '% of Account'
]

def map_and_extract(df, portfolio_name, original_col_order):
    # Standardize column names based on the mapping
    rename_dict = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in COLUMN_MAP:
            rename_dict[col] = COLUMN_MAP[col_stripped]
    df = df.rename(columns=rename_dict)
    for col in KEY_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[KEY_COLS]
    df['Portfolio Name'] = portfolio_name
    important_cols = ['Symbol', 'Description', 'Quantity', 'Market Value']
    df = df.dropna(subset=important_cols, how='all')
    df = append_holding_info(df)
    df = df[~(df.isnull() | df.eq('')).all(axis=1)]
    if 'Quantity' in original_col_order and 'Market Value' in original_col_order:
        orig_q = pd.to_numeric(df['Quantity'], errors='coerce')
        orig_mv = pd.to_numeric(df['Market Value'], errors='coerce')
        for idx, (q, mv) in enumerate(zip(orig_q, orig_mv)):
            if pd.isnull(q) and pd.isnull(mv):
                continue
            if not (q is None or mv is None):
                # This is not actually checking for mismatches - corrected to just pass
                pass
    return df

def consolidate_portfolios(dfs, mm_interest_rates=None):
    if not dfs:
        return pd.DataFrame()
    
    # The rest of the function should NOT be indented under the if block
    
    # Gather all unique columns from all input DataFrames
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)
    all_columns = list(all_columns)
    # Ensure all DataFrames have all columns
    dfs_full = []
    for df in dfs:
        df_full = df.copy()
        for col in all_columns:
            if col not in df_full.columns:
                df_full[col] = ""
        df_full = df_full[all_columns]
        dfs_full.append(df_full)
    # Start with the first DataFrame
    combined = dfs_full[0].copy()
    for df in dfs_full[1:]:
        for _, row in df.iterrows():
            symbol = row['Symbol']
            match = (combined['Symbol'] == symbol)
            if match.any():
                idx = combined.index[match][0]
                # Merge numeric columns (sum)
                for col in all_columns:
                    if col in ['Quantity', 'Market Value', 'Cost Basis', 'Gain/Loss $']:
                        combined.at[idx, col] = pd.to_numeric(combined.at[idx, col], errors='coerce') + pd.to_numeric(row[col], errors='coerce')
                # For Monthly Income, sum all numeric values
                if 'Monthly Income' in all_columns:
                    prev_val = combined.at[idx, 'Monthly Income']
                    new_val = row['Monthly Income']
                    vals = []
                    for v in [prev_val, new_val]:
                        try:
                            v_num = float(str(v).replace('$','').replace(',',''))
                            if not pd.isnull(v_num):
                                vals.append(v_num)
                        except:
                            pass
                    if vals:
                        combined.at[idx, 'Monthly Income'] = sum(vals)
                    else:
                        combined.at[idx, 'Monthly Income'] = ''
                # For Price, use weighted average
                if 'Price' in all_columns and 'Quantity' in all_columns:
                    total_qty = pd.to_numeric(combined.at[idx, 'Quantity'], errors='coerce')
                    prev_qty = total_qty - pd.to_numeric(row['Quantity'], errors='coerce')
                    if total_qty > 0:
                        combined.at[idx, 'Price'] = (
                            pd.to_numeric(combined.at[idx, 'Price'], errors='coerce') * prev_qty + pd.to_numeric(row['Price'], errors='coerce') * pd.to_numeric(row['Quantity'], errors='coerce')
                        ) / total_qty
                # For non-numeric columns, set to 'multiple' if values differ (except Monthly Income)
                for col in all_columns:
                    if col not in ['Symbol', 'Quantity', 'Market Value', 'Cost Basis', 'Gain/Loss $', 'Price', 'Monthly Income']:
                        if str(combined.at[idx, col]).strip() != str(row[col]).strip():
                            combined.at[idx, col] = 'multiple'
                # Source Portfolio
                if 'Portfolio Name' in all_columns:
                    combined.at[idx, 'Portfolio Name'] = 'multiple'
            else:
                # Add new row, ensure all columns present
                new_row = {}
                for col in all_columns:
                    new_row[col] = row[col] if col in row else ""
                combined = pd.concat([combined, pd.DataFrame([new_row])], ignore_index=True)
    # Recalculate % of Account
    if 'Market Value' in all_columns and '% of Account' in all_columns:
        total_mv = pd.to_numeric(combined['Market Value'], errors='coerce').sum()
        if total_mv > 0:
            # Calculate percentage and ensure it's stored as numeric
            combined['% of Account'] = pd.to_numeric(combined['Market Value'], errors='coerce') / total_mv * 100
            # Convert any non-numeric values to 0
            combined['% of Account'] = pd.to_numeric(combined['% of Account'], errors='coerce').fillna(0)
    # Return with all columns, preserving order from first input, then extras
    ordered_cols = list(dfs[0].columns) + [col for col in all_columns if col not in dfs[0].columns]
    return combined[ordered_cols]

def generate_portfolio_summary(dfs, consolidated_df, mm_interest_rates=None):
    """
    Generate a summary of the consolidated portfolio data
    
    Args:
        dfs: List of individual portfolio dataframes
        consolidated_df: Consolidated portfolio dataframe
        mm_interest_rates: Dictionary of money market interest rates by portfolio name
        
    Returns:
        Dictionary of summary information
    """
    summary = {}
    
    # 1. Overall Portfolio Statistics
    summary['num_portfolios'] = len(dfs)
    summary['total_holdings'] = len(consolidated_df)
    
    # Calculate total market value and cost basis
    total_market_value = pd.to_numeric(consolidated_df['Market Value'], errors='coerce').sum()
    summary['total_market_value'] = total_market_value
    
    total_cost_basis = pd.to_numeric(consolidated_df['Cost Basis'], errors='coerce').sum()
    summary['total_cost_basis'] = total_cost_basis
    
    # Calculate total gain/loss
    total_gain_loss = pd.to_numeric(consolidated_df['Gain/Loss $'], errors='coerce').sum()
    summary['total_gain_loss'] = total_gain_loss
    
    if total_cost_basis > 0:
        summary['total_gain_loss_percent'] = (total_gain_loss / total_cost_basis) * 100
    else:
        summary['total_gain_loss_percent'] = 0
    
    # Calculate total monthly income
    if 'Monthly Income' in consolidated_df.columns:
        monthly_income = pd.to_numeric(consolidated_df['Monthly Income'], errors='coerce').sum()
        summary['total_monthly_income'] = monthly_income
    else:
        summary['total_monthly_income'] = 0
    
    # 2. Individual Portfolio Breakdown
    portfolio_breakdown = []
    for df, name in zip(dfs, [df['Portfolio Name'].iloc[0] for df in dfs]):
        portfolio_market_value = pd.to_numeric(df['Market Value'], errors='coerce').sum()
        portfolio_cost_basis = pd.to_numeric(df['Cost Basis'], errors='coerce').sum()
        portfolio_gain_loss = pd.to_numeric(df['Gain/Loss $'], errors='coerce').sum()
        
        portfolio_data = {
            'name': name,
            'market_value': portfolio_market_value,
            'cost_basis': portfolio_cost_basis,
            'gain_loss': portfolio_gain_loss,
            'percent_of_total': (portfolio_market_value / total_market_value * 100) if total_market_value > 0 else 0,
            'num_holdings': len(df)
        }
        
        # Calculate portfolio monthly income if applicable
        if 'Monthly Income' in df.columns:
            portfolio_data['monthly_income'] = pd.to_numeric(df['Monthly Income'], errors='coerce').sum()
        
        portfolio_breakdown.append(portfolio_data)
    
    summary['portfolio_breakdown'] = portfolio_breakdown
    
    # 3. Asset Type Allocation
    if 'Type' in consolidated_df.columns or 'Holding Type' in consolidated_df.columns:
        # Determine which column to use for type breakdown
        type_column = 'Holding Type' if 'Holding Type' in consolidated_df.columns else 'Type'
        
        # Group by type and calculate market value and percentages
        type_groups = consolidated_df.groupby(type_column)
        
        asset_types = []
        for type_name, group in type_groups:
            type_market_value = pd.to_numeric(group['Market Value'], errors='coerce').sum()
            type_cost_basis = pd.to_numeric(group['Cost Basis'], errors='coerce').sum()
            type_gain_loss = pd.to_numeric(group['Gain/Loss $'], errors='coerce').sum()
            
            asset_type = {
                'type': type_name,
                'market_value': type_market_value,
                'cost_basis': type_cost_basis,
                'gain_loss': type_gain_loss,
                'percent_of_total': (type_market_value / total_market_value * 100) if total_market_value > 0 else 0,
                'num_holdings': len(group)
            }
            
            # Calculate type monthly income if applicable
            if 'Monthly Income' in group.columns:
                asset_type['monthly_income'] = pd.to_numeric(group['Monthly Income'], errors='coerce').sum()
            
            asset_types.append(asset_type)
        
        summary['asset_types'] = asset_types
    
    # 4. Top Holdings
    # Calculate top 10 holdings by market value
    if not consolidated_df.empty:
        top_holdings = consolidated_df.sort_values(by='Market Value', ascending=False).head(10)
        summary['top_holdings'] = top_holdings.to_dict('records')
    
    # 5. Income Summary
    if 'Monthly Income' in consolidated_df.columns:
        # Top 10 income-generating holdings
        top_income = consolidated_df.sort_values(by='Monthly Income', ascending=False).head(10)
        summary['top_income_holdings'] = top_income.to_dict('records')
        
        # Calculate portfolio yield
        annual_income = summary['total_monthly_income'] * 12
        if total_market_value > 0:
            summary['portfolio_yield'] = (annual_income / total_market_value) * 100
        else:
            summary['portfolio_yield'] = 0
    
    return summary

class PortfolioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Tabulator")
        self.root.geometry("700x700")  # Increased width and height for better display
        self.root.configure(padx=20, pady=20)
        
        # Default to downloads directory
        self.downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        
        self.input_files = []
        self.portfolio_names = []
        self.mm_interest_rates = {}  # Dictionary to store money market interest rates for each portfolio
        self.output_dir = self.downloads_dir
        self.progress_win = None
        self.progress_text = None
        self.create_widgets()

    def create_widgets(self):
        # Add title label with larger font
        from tkinter import font
        title_font = font.Font(size=14, weight="bold")
        title_label = Label(self.root, text="Portfolio Tabulator", font=title_font)
        title_label.pack(pady=10)
        
        # File selection section with frame
        files_frame = Frame(self.root, relief="groove", padx=10, pady=10, bd=1)
        files_frame.pack(fill="both", expand=True, pady=10)
        
        self.select_files_btn = Button(files_frame, text="Select Portfolio Files", 
                                      command=self.select_files, 
                                      bg="#4CAF50", fg="white", 
                                      padx=10, pady=5)
        self.select_files_btn.pack(pady=5)
        
        self.files_label = Label(files_frame, text="No files selected.")
        self.files_label.pack(pady=5)
        
        # Create a proper scrollable text area for files list instead of a label
        self.files_list_frame = Frame(files_frame)
        self.files_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.files_list = ScrolledText(self.files_list_frame, wrap="word", width=50, height=8)
        self.files_list.pack(fill="both", expand=True)
        self.files_list.config(state="disabled")  # Make it read-only
        
        # Output directory section with frame
        output_frame = Frame(self.root, relief="groove", padx=10, pady=10, bd=1)
        output_frame.pack(fill="x", pady=10)
        
        self.select_output_btn = Button(output_frame, text="Select Output Directory", 
                                       command=self.select_output_dir,
                                       bg="#2196F3", fg="white",
                                       padx=10, pady=5)
        self.select_output_btn.pack(pady=5)
        
        self.output_label = Label(output_frame, text=f"Output directory: {self.output_dir}")
        self.output_label.pack(pady=5)
        
        # Run button with better styling
        self.run_btn = Button(self.root, text="Run Processing", 
                             command=self.run_processing,
                             bg="#FF9800", fg="white",
                             padx=20, pady=10,
                             font=font.Font(size=12))
        self.run_btn.pack(pady=10)
        
        # About button
        self.about_btn = Button(self.root, text="About", 
                             command=self.show_about,
                             bg="#607D8B", fg="white",
                             padx=15, pady=5,
                             font=font.Font(size=10))
        self.about_btn.pack(pady=10)

    def show_about(self):
        about_text = """Portfolio Tabulator

This application processes portfolio data from Charles Schwab and Fidelity brokerage account downloads.

Inputs:
- Excel or CSV files exported from Charles Schwab or Fidelity accounts
- Each file should contain position details for a single portfolio

Output:
- Consolidated Excel workbook with detailed portfolio analysis
- Summary of all portfolios including market values, gain/loss, and income statistics
"""
        messagebox.showinfo("About Portfolio Tabulator", about_text)

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select Portfolio Files",
            filetypes=[("Portfolio files", "*.csv *.xlsx *.xls")]
        )
        if files:
            self.input_files = list(files)
            self.portfolio_names = []
            self.mm_interest_rates = {}  # Reset interest rates dictionary
            
            # Build a list of files to display in the UI
            files_display = []
            
            for file_path in self.input_files:
                base_name = os.path.basename(file_path)
                default_name = os.path.splitext(base_name)[0][:25]
                
                # Create custom dialog with focus and transient properties
                name_dialog_window = Toplevel(self.root)
                name_dialog_window.title("Portfolio Name")
                name_dialog_window.transient(self.root)  # Set as transient to main window
                name_dialog_window.grab_set()  # Make dialog modal
                name_dialog_window.geometry("400x150")  # Set reasonable size
                name_dialog_window.lift()  # Bring to front
                name_dialog_window.focus_force()  # Force focus
                name_dialog_window.attributes("-topmost", True)  # Keep on top
                
                Label(name_dialog_window, text=f"Enter a short name for this portfolio (default: {default_name}):", 
                      wraplength=350, pady=10).pack()
                
                entry_var = StringVar()
                entry = Entry(name_dialog_window, textvariable=entry_var, width=40)
                entry.pack(pady=10, padx=20)
                entry.focus_set()  # Set focus to entry
                
                result = [None]  # To store the result
                
                def on_ok():
                    result[0] = entry_var.get()
                    name_dialog_window.destroy()
                
                def on_cancel():
                    name_dialog_window.destroy()
                
                # Buttons frame
                button_frame = Frame(name_dialog_window)
                button_frame.pack(pady=10)
                
                Button(button_frame, text="OK", command=on_ok, width=10).pack(side=LEFT, padx=10)
                Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side=LEFT)
                
                # Center the dialog on the main window
                name_dialog_window.update_idletasks()
                x = self.root.winfo_x() + (self.root.winfo_width() - name_dialog_window.winfo_width()) // 2
                y = self.root.winfo_y() + (self.root.winfo_height() - name_dialog_window.winfo_height()) // 2
                name_dialog_window.geometry(f"+{x}+{y}")
                
                # Wait for the dialog to be closed
                self.root.wait_window(name_dialog_window)
                
                portfolio_name = result[0] or default_name
                self.portfolio_names.append(portfolio_name)
                
                # Ask for money market interest rate for this portfolio using the same custom dialog approach
                default_rate = "4.0"  # Default 4%
                rate_dialog_window = Toplevel(self.root)
                rate_dialog_window.title("Money Market Interest Rate")
                rate_dialog_window.transient(self.root)  # Set as transient to main window
                rate_dialog_window.grab_set()  # Make dialog modal
                rate_dialog_window.geometry("450x150")  # Set reasonable size
                rate_dialog_window.lift()  # Bring to front
                rate_dialog_window.focus_force()  # Force focus
                rate_dialog_window.attributes("-topmost", True)  # Keep on top
                
                Label(rate_dialog_window, 
                      text=f"Enter the money market interest rate for '{portfolio_name}' (default: {default_rate}%):", 
                      wraplength=400, pady=10).pack()
                
                rate_var = StringVar(value=default_rate)
                rate_entry = Entry(rate_dialog_window, textvariable=rate_var, width=10)
                rate_entry.pack(pady=10)
                rate_entry.focus_set()  # Set focus to entry
                rate_entry.select_range(0, len(default_rate))  # Select all text
                
                rate_result = [None]  # To store the result
                
                def on_rate_ok():
                    rate_result[0] = rate_var.get()
                    rate_dialog_window.destroy()
                
                def on_rate_cancel():
                    rate_dialog_window.destroy()
                
                # Buttons frame
                rate_button_frame = Frame(rate_dialog_window)
                rate_button_frame.pack(pady=10)
                
                Button(rate_button_frame, text="OK", command=on_rate_ok, width=10).pack(side=LEFT, padx=10)
                Button(rate_button_frame, text="Cancel", command=on_rate_cancel, width=10).pack(side=LEFT)
                
                # Center the dialog on the main window
                rate_dialog_window.update_idletasks()
                x = self.root.winfo_x() + (self.root.winfo_width() - rate_dialog_window.winfo_width()) // 2
                y = self.root.winfo_y() + (self.root.winfo_height() - rate_dialog_window.winfo_height()) // 2
                rate_dialog_window.geometry(f"+{x}+{y}")
                
                # Wait for the dialog to be closed
                self.root.wait_window(rate_dialog_window)
                
                mm_rate_str = rate_result[0] or default_rate
                
                # Convert to float and store as decimal (0.04 for 4%)
                try:
                    mm_rate = float(mm_rate_str.replace('%', '')) / 100
                    self.mm_interest_rates[portfolio_name] = mm_rate
                    files_display.append(f"• {base_name} as '{portfolio_name}' (MM Rate: {mm_rate*100:.2f}%)")
                except ValueError:
                    # If conversion fails, use default
                    self.mm_interest_rates[portfolio_name] = 0.04
                    files_display.append(f"• {base_name} as '{portfolio_name}' (MM Rate: 4.00%)")
            
            self.files_label.config(text=f"{len(self.input_files)} files selected:")
            
            # Update the scrollable files list
            self.files_list.config(state="normal")  # Enable editing to update
            self.files_list.delete(1.0, "end")      # Clear current content
            self.files_list.insert("end", "\n".join(files_display))
            self.files_list.config(state="disabled")  # Return to read-only
        else:
            self.input_files = []
            self.portfolio_names = []
            self.mm_interest_rates = {}
            self.files_label.config(text="No files selected.")
            self.files_list.config(state="normal")
            self.files_list.delete(1.0, "end")
            self.files_list.config(state="disabled")

    def select_output_dir(self):
        dir_path = filedialog.askdirectory(title="Choose Output Directory", initialdir=self.downloads_dir)
        if dir_path:
            self.output_dir = dir_path
            self.output_label.config(text=f"Output directory: {self.output_dir}")
        else:
            # If user cancels, keep the previous directory
            if not self.output_dir:
                self.output_dir = self.downloads_dir
            self.output_label.config(text=f"Output directory: {self.output_dir}")

    def run_processing(self):
        if not self.input_files:
            messagebox.showerror("Error", "Please select at least one input file.")
            return
        if not self.output_dir:
            # Default to downloads if somehow output_dir is still empty
            self.output_dir = self.downloads_dir
        
        print(f"Using output directory: {self.output_dir}")  # Debug print
            
        self.progress_win = Toplevel(self.root)
        self.progress_win.title("Progress / Debug Messages")
        self.progress_win.geometry("700x500")
        self.progress_win.transient(self.root)  # Set as transient to main window
        self.progress_win.lift()  # Bring to front
        self.progress_win.focus_force()  # Force focus
        
        # Position the progress window relative to the main window
        self.progress_win.update_idletasks()
        x = self.root.winfo_x() + 50
        y = self.root.winfo_y() + 50
        self.progress_win.geometry(f"+{x}+{y}")
        
        # Add a title to the progress window
        progress_title = Label(self.progress_win, text="Processing Files...", font=("Arial", 12, "bold"))
        progress_title.pack(pady=5)
        
        self.progress_text = ScrolledText(self.progress_win, width=80, height=24, state='disabled')
        self.progress_text.pack(padx=10, pady=10, fill="both", expand=True)
        
        threading.Thread(target=self.process_files, daemon=True).start()

    def log(self, msg):
        self.progress_text.config(state='normal')
        self.progress_text.insert('end', msg + "\n")
        self.progress_text.see('end')
        self.progress_text.config(state='disabled')
        self.progress_win.update()
        # Print directly to original stdout to avoid recursion
        sys.__stdout__.write(msg + "\n")
        sys.__stdout__.flush()

    def process_files(self):
        try:
            # Helper function to safely write numeric values to Excel
            def safe_write_number(worksheet, row, col, val, format):
                try:
                    if pd.isnull(val) or val == '':
                        worksheet.write(row, col, '', format)
                    else:
                        # Try to convert to float
                        float_val = float(val)
                        worksheet.write_number(row, col, float_val, format)
                except (ValueError, TypeError):
                    # If conversion fails, write as is
                    worksheet.write(row, col, val, format)
            
            downloads = os.path.join(os.path.expanduser("~"), "Downloads")
            outdir = self.output_dir if self.output_dir else downloads
            self.log(f"Using output directory: {outdir}")  # Debug log of actual output directory
            output_path = os.path.join(outdir, "processed_portfolio.xlsx")
            self.log(f"Will save output to: {output_path}")  # More debug information
            dfs = []
            sheet_names = []
            for file_path, portfolio_name in zip(self.input_files, self.portfolio_names):
                self.log(f"Processing: {file_path}")
                ext = os.path.splitext(file_path)[-1].lower()
                if ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(file_path)
                elif ext == ".csv":
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        first_line = f.readline()
                        if "Symbol" not in first_line and "Ticker" not in first_line:
                            skip = 3
                        else:
                            skip = 0
                    df = pd.read_csv(file_path, skiprows=skip)
                else:
                    self.log(f"Unsupported file type: {file_path}")
                    continue
                orig_cols = df.columns.tolist()
                df_proc = map_and_extract(df, portfolio_name, orig_cols)
                if df_proc.empty:
                    self.log(f"No valid data found in {file_path}")
                    continue
                dfs.append(df_proc)
                sheet_names.append(portfolio_name[:31])
            if not dfs:
                self.log("No valid data to save. Exiting.")
                return
                
            # Helper function to format cells correctly using write_number for numeric values
            def format_excel_data(df, worksheet, workbook, sheet_name):
                white_bg_format = workbook.add_format({'bg_color': '#FFFFFF'})
                header_format = workbook.add_format({'bg_color': '#FFFFFF', 'bold': True})
                
                # Define column formats
                currency_cols = ['Price', 'Market Value', 'Cost Basis', 'Gain/Loss $', 'Monthly Income']
                currency_format = workbook.add_format({
                    'num_format': '$#,##0.00', 
                    'bg_color': '#FFFFFF',
                    'align': 'right'
                })
                
                percent_format = workbook.add_format({
                    'num_format': '0.00%', 
                    'bg_color': '#FFFFFF',
                    'align': 'right'
                })
                
                quantity_format = workbook.add_format({
                    'num_format': '0.00', 
                    'bg_color': '#FFFFFF',
                    'align': 'right'
                })
                
                col_names = df.columns.tolist()
                
                # Set column widths and default formats
                for i, col in enumerate(col_names):
                    if col in currency_cols:
                        worksheet.set_column(i, i, 16, currency_format)
                    elif col == 'Quantity':
                        worksheet.set_column(i, i, 14, quantity_format)
                    elif col == 'Interest/Dividend':
                        worksheet.set_column(i, i, 12, percent_format)
                    elif col == '% of Account':
                        percent2_format = workbook.add_format({
                            'num_format': '0.00%', 
                            'bg_color': '#FFFFFF',
                            'align': 'right'
                        })
                        worksheet.set_column(i, i, 12, percent2_format)
                    else:
                        worksheet.set_column(i, i, max(14, len(str(col))+2), white_bg_format)
                
                # Write header row
                for j, col in enumerate(col_names):
                    worksheet.write(0, j, col, header_format)
                
                # Write data rows with proper formatting
                nrows, ncols = df.shape
                for i in range(nrows):
                    for j in range(ncols):
                        val = df.iloc[i, j]
                        col_name = col_names[j]
                        
                        # Handle NaN/inf values
                        if pd.isnull(val) or (isinstance(val, float) and (pd.isna(val) or np.isinf(val))):
                            worksheet.write(i+1, j, '', white_bg_format)
                        # Use safe_write_number for numeric columns with specific formats
                        elif col_name in currency_cols:
                            safe_write_number(worksheet, i+1, j, val, currency_format)
                        elif col_name == 'Quantity':
                            safe_write_number(worksheet, i+1, j, val, quantity_format)
                        elif col_name == 'Interest/Dividend':
                            safe_write_number(worksheet, i+1, j, val, percent_format)
                        elif col_name == '% of Account':
                            try:
                                if pd.isnull(val) or val == '':
                                    worksheet.write(i+1, j, '', white_bg_format)
                                else:
                                    # Convert to float to ensure division works
                                    float_val = float(val)
                                    worksheet.write_number(i+1, j, float_val/100, percent2_format)
                            except (ValueError, TypeError):
                                # Fall back to regular write if conversion fails
                                worksheet.write(i+1, j, val, white_bg_format)
                        else:
                            worksheet.write(i+1, j, val, white_bg_format)
            
            self.log("Saving all sheets to Excel...")
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                for df, name in zip(dfs, sheet_names):
                    # Format dollar amounts and quantities as numeric values but don't pre-format
                    # This allows Excel to handle the formatting and enables summing
                    currency_cols = ['Price', 'Market Value', 'Cost Basis', 'Gain/Loss $', 'Monthly Income']
                    for col in currency_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Format quantity with two decimal places
                    if 'Quantity' in df.columns:
                        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
                    
                    df.to_excel(writer, sheet_name=name, index=False)
                    workbook = writer.book
                    worksheet = writer.sheets[name]
                    white_bg_format = workbook.add_format({'bg_color': '#FFFFFF'})
                    header_format = workbook.add_format({'bg_color': '#FFFFFF', 'bold': True})
                    col_names = df.columns.tolist()
                    
                    # Create better formats with stronger right-alignment and wider columns
                    currency_cols = ['Price', 'Market Value', 'Cost Basis', 'Gain/Loss $', 'Monthly Income']
                    currency_format = workbook.add_format({
                        'num_format': '$#,##0.00', 
                        'bg_color': '#FFFFFF',
                        'align': 'right'
                    })
                    
                    # Apply formats to all numeric columns
                    for col in currency_cols:
                        if col in col_names:
                            idx = col_names.index(col)
                            # Make currency columns wider for better visibility
                            worksheet.set_column(idx, idx, 16, currency_format)
                    
                    # Special formats for specific column types
                    if 'Interest/Dividend' in col_names:
                        idx = col_names.index('Interest/Dividend')
                        percent_format = workbook.add_format({
                            'num_format': '0.00%', 
                            'bg_color': '#FFFFFF',
                            'align': 'right'
                        })
                        worksheet.set_column(idx, idx, 12, percent_format)
                    
                    # Format Quantity with 2 decimal places and right alignment
                    if 'Quantity' in col_names:
                        idx = col_names.index('Quantity')
                        quantity_format = workbook.add_format({
                            'num_format': '0.00', 
                            'bg_color': '#FFFFFF',
                            'align': 'right'
                        })
                        worksheet.set_column(idx, idx, 14, quantity_format)
                    
                    for i, col in enumerate(col_names):
                        if col not in currency_cols + ['Interest/Dividend']:
                            worksheet.set_column(i, i, max(14, len(str(col))+2), white_bg_format)
                    # Overwrite all cells (header and data) with white background
                    nrows, ncols = df.shape
                    # Header row
                    for j, col in enumerate(col_names):
                        worksheet.write(0, j, col, header_format)
                    # Data rows
                    for i in range(nrows):
                        for j in range(ncols):
                            val = df.iloc[i, j]
                            col_name = col_names[j]
                            
                            # Replace NaN/inf with empty string
                            if pd.isnull(val) or (isinstance(val, float) and (pd.isna(val) or np.isinf(val))):
                                worksheet.write(i+1, j, '', white_bg_format)
                            # Use the specific format for each column type
                            elif col_name in currency_cols:
                                safe_write_number(worksheet, i+1, j, val, currency_format)
                            elif col_name == 'Quantity':
                                safe_write_number(worksheet, i+1, j, val, quantity_format)
                            elif col_name == 'Interest/Dividend':
                                safe_write_number(worksheet, i+1, j, val, percent_format)
                            elif col_name == '% of Account':
                                try:
                                    if pd.isnull(val) or val == '':
                                        worksheet.write(i+1, j, '', white_bg_format)
                                    else:
                                        # Convert to float to ensure division works
                                        float_val = float(val)
                                        worksheet.write_number(i+1, j, float_val/100, percent2_format)
                                except (ValueError, TypeError):
                                    # Fall back to regular write if conversion fails
                                    worksheet.write(i+1, j, val, white_bg_format)
                            else:
                                worksheet.write(i+1, j, val, white_bg_format)
                # Add consolidated tab
                consolidated_df = consolidate_portfolios(dfs, self.mm_interest_rates)
                if not consolidated_df.empty:
                    # Set Interest/Dividend for Money Market holdings and calculate Monthly Income
                    if 'Interest/Dividend' not in consolidated_df.columns:
                        consolidated_df['Interest/Dividend'] = ""
                    
                    # Ensure Monthly Income column exists
                    if 'Monthly Income' not in consolidated_df.columns:
                        consolidated_df['Monthly Income'] = ""
                    
                    # Process Money Market holdings
                    for idx, row in consolidated_df.iterrows():
                        # Check if this is a Money Market holding using both Type and Holding Type columns
                        type_val = str(row['Type']).strip().upper()
                        holding_type = ""
                        if 'Holding Type' in consolidated_df.columns:
                            holding_type = str(row['Holding Type']).strip().upper()
                        
                        # Only consider rows as money market if the "Holding Type" column is "Money Market" or "MONEYMARKET"
                        is_money_market = (
                            holding_type == 'MONEY MARKET' or 
                            holding_type == 'MONEYMARKET'
                        )
                        
                        if is_money_market:
                            self.log(f"Found Money Market: {row['Symbol']} - {row['Description']}")
                            # Get the interest rate (either existing or set a new one)
                            current_rate = pd.to_numeric(row['Interest/Dividend'], errors='coerce')
                            rate = current_rate  # Default to using existing rate
                            
                            # If Interest/Dividend not set, use portfolio-specific or default rate
                            if pd.isna(current_rate) or current_rate == 0:
                                portfolio_name = row['Portfolio Name']
                                if portfolio_name in self.mm_interest_rates:
                                    rate = self.mm_interest_rates[portfolio_name]
                                else:
                                    rate = 0.04  # 4% default
                                consolidated_df.at[idx, 'Interest/Dividend'] = rate
                                self.log(f"Set interest rate for {row['Symbol']} to {rate*100:.2f}%")
                            
                            # Calculate monthly income for all money market holdings
                            market_value = pd.to_numeric(row['Market Value'], errors='coerce')
                            if not pd.isna(market_value) and market_value > 0:
                                # Calculate monthly income (annual rate / 12 * market value)
                                monthly_income = (rate / 12) * market_value
                                consolidated_df.at[idx, 'Monthly Income'] = monthly_income
                                self.log(f"Set Monthly Income for {row['Symbol']} to ${monthly_income:.2f} based on {rate*100:.2f}% interest rate")
                    
                    self.log("Money Market holdings processed with interest rates and monthly income calculations")
                    
                    # Convert columns to numeric values but don't pre-format as strings
                    # This lets Excel handle the formatting and allows summing
                    currency_cols = ['Price', 'Market Value', 'Cost Basis', 'Gain/Loss $', 'Monthly Income']
                    for col in currency_cols:
                        if col in consolidated_df.columns:
                            consolidated_df[col] = pd.to_numeric(consolidated_df[col], errors='coerce')
                    
                    # Format quantity with two decimal places
                    if 'Quantity' in consolidated_df.columns:
                        consolidated_df['Quantity'] = pd.to_numeric(consolidated_df['Quantity'], errors='coerce')
                    
                # Format '% of Account' as percentage with 2 decimal points
                    if '% of Account' in consolidated_df.columns:
                        try:
                            # Ensure it's properly converted to numeric values
                            consolidated_df['% of Account'] = pd.to_numeric(consolidated_df['% of Account'], errors='coerce').fillna(0).round(2)
                            self.log("Converted '% of Account' column to numeric values")
                        except Exception as e:
                            self.log(f"Warning: Could not convert '% of Account' column to numeric: {str(e)}")
                    
                    # Ensure all numeric columns are properly converted to numeric types
                    for col in currency_cols + ['Quantity']:
                        if col in consolidated_df.columns:
                            consolidated_df[col] = pd.to_numeric(consolidated_df[col], errors='coerce').fillna(0)
                    
                    # Ensure Interest/Dividend is properly formatted as a percentage
                    if 'Interest/Dividend' in consolidated_df.columns:
                        consolidated_df['Interest/Dividend'] = pd.to_numeric(consolidated_df['Interest/Dividend'], errors='coerce').fillna(0)
                    
                    consolidated_df.to_excel(writer, sheet_name='Consolidated Holdings', index=False)
                    workbook = writer.book
                    worksheet = writer.sheets['Consolidated Holdings']
                    
                    # Use our formatting function
                    self.log("Formatting consolidated sheet...")
                    
                    # Define a simple function for safe conversion
                    def is_numeric(val):
                        if isinstance(val, (int, float)):
                            return True
                        if isinstance(val, str):
                            try:
                                float(val.replace('%', '').replace('$', '').replace(',', ''))
                                return True
                            except (ValueError, TypeError):
                                return False
                        return False
                    
                    # Add safe conversion of percentage values
                    for i in range(len(consolidated_df)):
                        if '% of Account' in consolidated_df.columns:
                            val = consolidated_df.at[i, '% of Account']
                            if not is_numeric(val):
                                self.log(f"Warning: Non-numeric value in % of Account: {val}")
                                # Try to convert it to a valid number or 0
                                consolidated_df.at[i, '% of Account'] = 0
                    
                    try:
                        format_excel_data(consolidated_df, worksheet, workbook, 'Consolidated Holdings')
                        self.log("Consolidated sheet formatted successfully")
                    except Exception as e:
                        self.log(f"Error during Excel formatting: {str(e)}")
                        # Print the problematic data
                        for col in consolidated_df.columns:
                            if col == '% of Account':
                                self.log(f"Column '{col}' data types: {consolidated_df[col].apply(type).value_counts()}")
                    
                    # Apply formats to all numeric columns
                    for col in currency_cols:
                        if col in col_names:
                            idx = col_names.index(col)
                            # Make currency columns wider for better visibility
                            worksheet.set_column(idx, idx, 16, currency_format)
                    
                    # Special formats for specific column types
                    if 'Interest/Dividend' in col_names:
                        idx = col_names.index('Interest/Dividend')
                        percent_format = workbook.add_format({
                            'num_format': '0.00%', 
                            'bg_color': '#FFFFFF',
                            'align': 'right'
                        })
                        worksheet.set_column(idx, idx, 12, percent_format)
                    
                    # Set '% of Account' column to percentage format with white background
                    if '% of Account' in col_names:
                        idx = col_names.index('% of Account')
                        percent2_format = workbook.add_format({
                            'num_format': '0.00%', 
                            'bg_color': '#FFFFFF',
                            'align': 'right'
                        })
                        worksheet.set_column(idx, idx, 12, percent2_format)
                        
                    # Format Quantity with 2 decimal places and right alignment
                    if 'Quantity' in col_names:
                        idx = col_names.index('Quantity')
                        quantity_format = workbook.add_format({
                            'num_format': '0.00', 
                            'bg_color': '#FFFFFF',
                            'align': 'right'
                        })
                        worksheet.set_column(idx, idx, 14, quantity_format)
                    for i, col in enumerate(col_names):
                        if col not in currency_cols + ['Interest/Dividend', '% of Account']:
                            worksheet.set_column(i, i, max(14, len(str(col))+2), white_bg_format)
                    # Overwrite all cells (header and data) with white background
                    nrows, ncols = consolidated_df.shape
                    # Header row
                    for j, col in enumerate(col_names):
                        worksheet.write(0, j, col, header_format)
                    # Data rows
                    for i in range(nrows):
                        for j in range(ncols):
                            val = consolidated_df.iloc[i, j]
                            col_name = col_names[j]
                            
                            # Replace NaN/inf with empty string
                            if pd.isnull(val) or (isinstance(val, float) and (pd.isna(val) or np.isinf(val))):
                                worksheet.write(i+1, j, '', white_bg_format)
                            # Use the specific format for each column type
                            elif col_name in currency_cols:
                                safe_write_number(worksheet, i+1, j, val, currency_format)
                            elif col_name == 'Quantity':
                                safe_write_number(worksheet, i+1, j, val, quantity_format)
                            elif col_name == 'Interest/Dividend':
                                safe_write_number(worksheet, i+1, j, val, percent_format)
                            elif col_name == '% of Account':
                                try:
                                    if pd.isnull(val) or val == '':
                                        worksheet.write(i+1, j, '', white_bg_format)
                                    else:
                                        # Convert to float to ensure division works
                                        float_val = float(val)
                                        worksheet.write_number(i+1, j, float_val/100, percent2_format)
                                except (ValueError, TypeError):
                                    # Fall back to regular write if conversion fails
                                    worksheet.write(i+1, j, val, white_bg_format)
                            else:
                                worksheet.write(i+1, j, val, white_bg_format)
                    
                    # Generate and add Portfolio Summary tab
                    self.log("Creating Portfolio Summary tab...")
                    
                    # Generate summary data
                    summary_data = generate_portfolio_summary(dfs, consolidated_df, self.mm_interest_rates)
                    
                    # Create Summary worksheet
                    summary_sheet = writer.book.add_worksheet('Portfolio Summary')
                    
                    # Set column widths and white background for all columns
                    white_bg_format = workbook.add_format({'bg_color': '#FFFFFF'})
                    summary_sheet.set_column('A:A', 30, white_bg_format)  # Description column
                    summary_sheet.set_column('B:B', 20, white_bg_format)  # Value column
                    summary_sheet.set_column('C:C', 20, white_bg_format)  # Additional data column
                    
                    # Define formats
                    title_format = workbook.add_format({
                        'bold': True,
                        'font_size': 14,
                        'bg_color': '#FFFFFF',  # Changed to white background
                        'border': 1,
                        'align': 'center',
                        'valign': 'vcenter'
                    })
                    
                    header_format = workbook.add_format({
                        'bold': True,
                        'bg_color': '#FFFFFF',  # Changed to white background
                        'border': 1,
                        'align': 'left'
                    })
                    
                    subheader_format = workbook.add_format({
                        'bold': True,
                        'bg_color': '#FFFFFF',  # Changed to white background
                        'border': 1,
                        'align': 'left'
                    })
                    
                    data_format = workbook.add_format({
                        'border': 1,
                        'align': 'left',
                        'bg_color': '#FFFFFF'  # Added white background
                    })
                    
                    currency_format = workbook.add_format({
                        'num_format': '$#,##0.00',
                        'border': 1,
                        'align': 'right',
                        'bg_color': '#FFFFFF'  # Added white background
                    })
                    
                    percent_format = workbook.add_format({
                        'num_format': '0.00%',
                        'border': 1,
                        'align': 'right',
                        'bg_color': '#FFFFFF'  # Added white background
                    })
                    
                    count_format = workbook.add_format({
                        'num_format': '#,##0',
                        'border': 1,
                        'align': 'right',
                        'bg_color': '#FFFFFF'  # Added white background
                    })
                    
                    # Add title
                    summary_sheet.merge_range('A1:C1', 'CONSOLIDATED PORTFOLIO SUMMARY', title_format)
                    
                    # Add current date
                    from datetime import datetime
                    current_date = datetime.now().strftime("%B %d, %Y")
                    summary_sheet.merge_range('A2:C2', f'Generated on: {current_date}', header_format)
                    
                    # Section 1: Overall Portfolio Statistics
                    summary_sheet.merge_range('A4:C4', 'OVERALL PORTFOLIO STATISTICS', header_format)
                    
                    row = 4
                    # Number of portfolios
                    summary_sheet.write(row, 0, 'Number of Portfolios:', data_format)
                    summary_sheet.write(row, 1, summary_data['num_portfolios'], count_format)
                    row += 1
                    
                    # Total Holdings
                    summary_sheet.write(row, 0, 'Total Number of Holdings:', data_format)
                    summary_sheet.write(row, 1, summary_data['total_holdings'], count_format)
                    row += 1
                    
                    # Total Market Value
                    summary_sheet.write(row, 0, 'Total Market Value:', data_format)
                    summary_sheet.write(row, 1, summary_data['total_market_value'], currency_format)
                    row += 1
                    
                    # Total Cost Basis
                    summary_sheet.write(row, 0, 'Total Cost Basis:', data_format)
                    summary_sheet.write(row, 1, summary_data['total_cost_basis'], currency_format)
                    row += 1
                    
                    # Total Gain/Loss
                    summary_sheet.write(row, 0, 'Total Gain/Loss ($):', data_format)
                    summary_sheet.write(row, 1, summary_data['total_gain_loss'], currency_format)
                    row += 1
                    
                    # Total Gain/Loss Percent
                    summary_sheet.write(row, 0, 'Total Gain/Loss (%):', data_format)
                    summary_sheet.write(row, 1, summary_data['total_gain_loss_percent']/100, percent_format)
                    row += 1
                    
                    # Total Monthly Income
                    summary_sheet.write(row, 0, 'Total Monthly Income:', data_format)
                    summary_sheet.write(row, 1, summary_data['total_monthly_income'], currency_format)
                    row += 1
                    
                    # Annualized Income
                    summary_sheet.write(row, 0, 'Annualized Income:', data_format)
                    summary_sheet.write(row, 1, summary_data['total_monthly_income'] * 12, currency_format)
                    row += 1
                    
                    # Portfolio Yield
                    if 'portfolio_yield' in summary_data:
                        summary_sheet.write(row, 0, 'Portfolio Yield:', data_format)
                        summary_sheet.write(row, 1, summary_data['portfolio_yield']/100, percent_format)
                        row += 1
                    
                    # Section 2: Individual Portfolio Breakdown
                    row += 1
                    summary_sheet.merge_range(f'A{row+1}:C{row+1}', 'INDIVIDUAL PORTFOLIO BREAKDOWN', header_format)
                    row += 1
                    
                    # Column headers
                    summary_sheet.write(row, 0, 'Portfolio Name', subheader_format)
                    summary_sheet.write(row, 1, 'Market Value', subheader_format)
                    summary_sheet.write(row, 2, '% of Total', subheader_format)
                    row += 1
                    
                    # Portfolio data
                    for portfolio in summary_data['portfolio_breakdown']:
                        summary_sheet.write(row, 0, portfolio['name'], data_format)
                        summary_sheet.write(row, 1, portfolio['market_value'], currency_format)
                        summary_sheet.write(row, 2, portfolio['percent_of_total']/100, percent_format)
                        row += 1
                    
                    # Section 3: Asset Type Allocation
                    if 'asset_types' in summary_data:
                        row += 1
                        summary_sheet.merge_range(f'A{row+1}:C{row+1}', 'ASSET TYPE ALLOCATION', header_format)
                        row += 1
                        
                        # Column headers
                        summary_sheet.write(row, 0, 'Asset Type', subheader_format)
                        summary_sheet.write(row, 1, 'Market Value', subheader_format)
                        summary_sheet.write(row, 2, '% of Total', subheader_format)
                        row += 1
                        
                        # Asset type data
                        for asset_type in summary_data['asset_types']:
                            summary_sheet.write(row, 0, asset_type['type'], data_format)
                            summary_sheet.write(row, 1, asset_type['market_value'], currency_format)
                            summary_sheet.write(row, 2, asset_type['percent_of_total']/100, percent_format)
                            row += 1
                    
                    # Section 4: Top Holdings
                    if 'top_holdings' in summary_data:
                        row += 1
                        summary_sheet.merge_range(f'A{row+1}:C{row+1}', 'TOP 10 HOLDINGS BY VALUE', header_format)
                        row += 1
                        
                        # Column headers
                        summary_sheet.write(row, 0, 'Security', subheader_format)
                        summary_sheet.write(row, 1, 'Market Value', subheader_format)
                        summary_sheet.write(row, 2, '% of Total', subheader_format)
                        row += 1
                        
                        # Top holdings data
                        for holding in summary_data['top_holdings']:
                            holding_name = f"{holding['Symbol']} - {holding['Description']}"
                            market_value = pd.to_numeric(holding['Market Value'], errors='coerce')
                            percent = pd.to_numeric(holding['% of Account'], errors='coerce')
                            
                            summary_sheet.write(row, 0, holding_name, data_format)
                            summary_sheet.write(row, 1, market_value, currency_format)
                            summary_sheet.write(row, 2, percent/100, percent_format)
                            row += 1
                    
                    # Section 5: Top Income Holdings
                    if 'top_income_holdings' in summary_data:
                        row += 1
                        summary_sheet.merge_range(f'A{row+1}:C{row+1}', 'TOP 10 INCOME PRODUCING HOLDINGS', header_format)
                        row += 1
                        
                        # Column headers
                        summary_sheet.write(row, 0, 'Security', subheader_format)
                        summary_sheet.write(row, 1, 'Monthly Income', subheader_format)
                        summary_sheet.write(row, 2, 'Annual Income', subheader_format)
                        row += 1
                        
                        # Top income holdings data
                        for holding in summary_data['top_income_holdings']:
                            holding_name = f"{holding['Symbol']} - {holding['Description']}"
                            monthly_income = pd.to_numeric(holding['Monthly Income'], errors='coerce')
                            
                            summary_sheet.write(row, 0, holding_name, data_format)
                            summary_sheet.write(row, 1, monthly_income, currency_format)
                            summary_sheet.write(row, 2, monthly_income * 12, currency_format)
                            row += 1
                    
                    self.log("Portfolio Summary tab created successfully")
            
            self.log(f"Success! Output saved to {output_path}")
            messagebox.showinfo("Success!", f"Output saved to {output_path}")
            
            # Try to open the file automatically
            try:
                os.startfile(output_path)
            except:
                pass  # Silently fail if can't open
        except Exception as e:
            self.log(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    print("Starting Portfolio GUI...")
    # Redirect stdout and stderr to a StringIO object to capture print statements
    import sys
    from io import StringIO
    
    class PrintRedirector:
        def __init__(self):
            self.log_window = None
            self.recursion_guard = False
        
        def set_log_window(self, log_window):
            self.log_window = log_window
        
        def write(self, text):
            # Write to original stdout
            sys.__stdout__.write(text)
            
            # Only forward to log window if not in a recursive call and text is not empty
            if self.log_window and text.strip() and not self.recursion_guard:
                try:
                    self.recursion_guard = True  # Set guard to prevent recursion
                    self.log_window(text.strip())
                finally:
                    self.recursion_guard = False  # Reset guard after completion
        
        def flush(self):
            sys.__stdout__.flush()
    
    # Create redirector instance
    redirector = PrintRedirector()
    sys.stdout = redirector
    
    root = Tk()
    gui = PortfolioGUI(root)
    
    # Set the log window reference for the redirector
    def log_to_window(text):
        if gui.progress_text and hasattr(gui, 'log'):
            try:
                gui.log(text)
            except Exception as e:
                # In case of error, write directly to original stdout
                sys.__stdout__.write(f"Error in log_to_window: {str(e)}\n")
                sys.__stdout__.flush()
    
    redirector.set_log_window(log_to_window)
    
    root.mainloop()
