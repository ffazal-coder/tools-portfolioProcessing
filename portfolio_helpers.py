import yfinance as yf
import re
import time

import yfinance as yf
import pandas as pd
import numpy as np


def extract_cd_details(description):
    """
    Extract interest rate and maturity date from CD description.
    Returns (interest as string, maturity as string)
    """
    desc = description.upper()
    interest_match = re.search(r'(\d+\.\d+)%', desc)
    maturity_match = re.search(r'DUE\s+(\d{2}/\d{2}/\d{2,4}|\d{2}-\d{2}-\d{2,4}|\d{2}/\d{2}/\d{4})', desc)

    interest = ""
    maturity = ""

    if interest_match:
        try:
            interest_value = float(interest_match.group(1))
            interest = f"{interest_value:.2f}%"
        except:
            interest = interest_match.group(1) + '%'

    if maturity_match:
        maturity = maturity_match.group(1)

    return interest, maturity


def is_cd(desc):
    """
    Recognizes CDs by:
    - ' CD ' (spaces)
    - 'CD FDIC'
    """
    desc_upper = desc.upper()
    return " CD " in f" {desc_upper} " or "CD FDIC" in desc_upper

import re



def extract_treasury_details(desc):
    """
    Parses treasury description to extract interest rate, maturity date, and type.
    Example: "US TREASUR NT 4.5%07/26UST NOTE DUE 07/15/26"
    """
    details = {}

    # --- INTEREST RATE ---
    interest_match = re.search(r'\s(\d+\.\d+)%', desc)
    if interest_match:
        try:
            details['interest'] = round(float(interest_match.group(1)), 2)
            print(f"    [Debug] Extracted Treasury Interest Rate: {details['interest']}%")
        except ValueError:
            print(f"    [!] Failed to convert interest: {interest_match.group(1)}")
            details['interest'] = ''
    else:
        print(f"    [!] No Treasury interest match found in: {desc}")
        details['interest'] = ''

    # --- MATURITY DATE ---
    maturity_match = re.search(r'DUE\s+(\d{2}/\d{2}/\d{2,4})', desc.upper())
    if maturity_match:
        details['maturity'] = maturity_match.group(1)
        print(f"    [Debug] Extracted Treasury Maturity Date: {details['maturity']}")
    else:
        print(f"    [!] No Treasury maturity date found in: {desc}")
        details['maturity'] = ''

    # --- TYPE (NOTE, BILL, BOND) ---
    if 'BOND' in desc.upper():
        details['type'] = 'Bond'
    elif 'BILL' in desc.upper():
        details['type'] = 'Bill'
    elif 'NOTE' in desc.upper():
        details['type'] = 'Note'
    else:
        details['type'] = ''
        print(f"    [!] No Treasury type found in: {desc}")

    return details




def is_treasury(desc):
    """
    Return True if the description indicates a US Treasury holding.
    Requires both a whole-word match for TREASURY, TREASUR, or UST,
    and the word 'DUE' to appear in the description.
    """
    d = desc.upper()
    # Must match one of these words AND contain 'DUE'
    return bool(re.search(r'\b(TREASURY|TREASUR|UST)\b', d)) and 'DUE' in d



def clean_name(name):
    # Remove trailing non-alphanumerics (like *, **, etc.)
    return re.sub(r'[\W_]+$', '', name.strip())




def format_currency(val):
    if pd.isna(val) or val == '':
        return ''
    try:
        return '${:,.0f}'.format(round(float(str(val).replace(',','').replace('$',''))))
    except Exception:
        return val

def format_percent(val):
    if pd.isna(val) or val == '':
        return ''
    try:
        return '{:.2f}%'.format(float(str(val).replace('%','')))
    except Exception:
        return val

def compute_monthly_income(row):
    try:
        mv = float(str(row.get('Market Value', '')).replace(',','').replace('$',''))
        rate_str = str(row.get('Interest/Dividend', '')).replace('%','')
        rate = float(rate_str)/100 if rate_str not in ['', None] else 0.0
        if rate > 0:
            return round(mv * rate / 12, 2)
        else:
            return 0.0
    except Exception:
        return 0.0



def append_holding_info(df, symbol_col="Symbol", desc_col="Description"):
    # Ensure output columns
    for newcol in ['Holding Type', 'Dividend or Interest', 'Interest Rate', 'Maturity Date', 'Lookup Status']:
        if newcol not in df.columns:
            df[newcol] = ""
    rows_to_drop = []
    for idx, row in df.iterrows():
        symbol = str(row.get(symbol_col, "")).strip()
        desc = str(row.get(desc_col, "")).strip()

        # Strip trailing non-alphanumerics (superscripts, etc.) from symbol for lookup and output
        symbol_lookup = re.sub(r'[\W_]+$', '', symbol)
        # Also update the DataFrame so output Symbol column is cleaned
        df.at[idx, symbol_col] = symbol_lookup
        symbol_clean = symbol.lower().strip()
        desc_clean = desc.lower().strip()
        print(f"\nProcessing row {idx}: Symbol={symbol!r}, Description={desc!r}")

        # --- REMOVE "Account Total" or "Total Account" ROWS (in Symbol OR Description) ---
        account_total_keys = ["account total", "total account"]
        if any(
            desc_clean == key or desc_clean.startswith(key) or
            symbol_clean == key or symbol_clean.startswith(key)
            for key in account_total_keys
        ):
            print("  -> Dropping row: Account/Total detected")
            rows_to_drop.append(idx)
            continue

        # --- CASH LOGIC: (in Symbol OR Description) ---
        if (
            "cash" in desc_clean or
            "cash & cash investments" in desc_clean or
            "cash" in symbol_clean or
            "cash & cash investments" in symbol_clean or
            symbol.strip().upper() == "CASH"
        ):
            print("  -> Classified as: Cash")
            df.at[idx, 'Holding Type'] = "Cash"
            df.at[idx, 'Dividend or Interest'] = ""
            df.at[idx, 'Interest Rate'] = ""
            df.at[idx, 'Maturity Date'] = ""
            df.at[idx, 'Lookup Status'] = "Identified as Cash"
            continue

        # --- CDs ---
        if 'is_cd' in globals() and is_cd(desc):
            interest, maturity = extract_cd_details(desc)
            print(f"  -> Classified as: CD (Interest={interest}, Maturity={maturity})")
            df.at[idx, 'Holding Type'] = "CD"
            df.at[idx, 'Dividend or Interest'] = "Interest"
            df.at[idx, 'Interest Rate'] = interest
            df.at[idx, 'Maturity Date'] = maturity
            df.at[idx, 'Lookup Status'] = "Parsed from Description"
            continue

        # --- TREASURIES ---
        if is_treasury(desc):
            details = extract_treasury_details(desc)
            interest = details.get('interest', '')
            maturity = details.get('maturity', '')
            treasury_type = details.get('type', '')

            print(f"  -> Classified as: Treasury {treasury_type} (Interest={interest}, Maturity={maturity})")
            df.at[idx, 'Holding Type'] = f"Treasury {treasury_type}"
            df.at[idx, 'Dividend or Interest'] = "Interest"
            df.at[idx, 'Interest Rate'] = interest
            df.at[idx, 'Maturity Date'] = maturity
            df.at[idx, 'Lookup Status'] = "Parsed from Description"
            continue

        # --- MONEY MARKET ---
        if "money market" in desc_clean:
            print("  -> Classified as: Money Market")
            df.at[idx, 'Holding Type'] = "Money Market"
            df.at[idx, 'Dividend or Interest'] = "Interest"
            df.at[idx, 'Interest Rate'] = ""
            df.at[idx, 'Maturity Date'] = ""
            df.at[idx, 'Lookup Status'] = "Identified by Description"
            continue

        # --- yfinance for all others ---
        if symbol_lookup:
            try:
                ticker = yf.Ticker(symbol_lookup)
                info = ticker.info
                quote_type = info.get('quoteType', '')
                holding_type_map = {
                    'equity': 'Stock',
                    'etf': 'ETF',
                    'mutualfund': 'Mutual Fund',
                    'moneyMarket': 'Money Market',
                    'bond': 'Fixed Income',
                    'cd': 'CD'
                }
                holding_type = holding_type_map.get(quote_type.lower(), quote_type)
                # Gather all possible dividend/yield fields
                div_rate = info.get('dividendRate', '')
                div_yield = info.get('dividendYield', '')
                fund_yield = info.get('yield', '')
                last_div = info.get('lastDividendValue', '')

                dividend_or_interest = "Dividend" if holding_type in ['Stock', 'ETF', 'Mutual Fund'] else "Interest"
                maturity = info.get('maturityDate', '')
                print(f"  -> Classified by yfinance: {holding_type} (quoteType={quote_type})")
                print(f"     Dividend Rate: {div_rate}")
                print(f"     Dividend Yield: {div_yield}")
                print(f"     Fund Yield: {fund_yield}")
                print(f"     Last Dividend Value: {last_div}")
                # Set dividend/interest rate logic - always prefer dividendYield or fund_yield for percent column
                rate_val = ""
                if div_yield not in [None, '', 0, 0.0]:
                    try:
                        dy = float(div_yield)
                        rate_val = round(dy, 4)
                    except Exception:
                        rate_val = ""
                elif fund_yield not in [None, '', 0, 0.0]:
                    try:
                        fy = float(fund_yield)
                        rate_val = round(fy, 4)
                    except Exception:
                        rate_val = ""
                elif div_rate not in [None, '', 0, 0.0]:
                    price_yf = info.get('regularMarketPrice') or info.get('previousClose') or None
                    try:
                        price_local = float(row.get('Price', 0))
                        if not price_yf and price_local > 0:
                            price_yf = price_local
                    except Exception:
                        pass
                    if price_yf and float(price_yf) > 0:
                        rate_val = round(float(div_rate) / float(price_yf) * 100, 4)
                    else:
                        rate_val = ""
                elif last_div not in [None, '', 0, 0.0]:
                    rate_val = last_div
                else:
                    rate_val = ""

                df.at[idx, 'Holding Type'] = holding_type
                df.at[idx, 'Dividend or Interest'] = dividend_or_interest
                df.at[idx, 'Interest Rate'] = rate_val
                df.at[idx, 'Maturity Date'] = maturity if maturity else ""
                df.at[idx, 'Lookup Status'] = "yfinance OK"
            except Exception as e:
                print(f"  -> yfinance failed: {e} | Fallback to Unknown")
                df.at[idx, 'Holding Type'] = "Unknown"
                df.at[idx, 'Dividend or Interest'] = "Unknown"
                df.at[idx, 'Interest Rate'] = ""
                df.at[idx, 'Maturity Date'] = ""
                df.at[idx, 'Lookup Status'] = f"yfinance failed: {e}"
            time.sleep(0.5)
        else:
            print("  -> Classified as: Unknown (no symbol or fallback)")
            df.at[idx, 'Holding Type'] = "Unknown"
            df.at[idx, 'Dividend or Interest'] = "Unknown"
            df.at[idx, 'Interest Rate'] = ""
            df.at[idx, 'Maturity Date'] = ""
            df.at[idx, 'Lookup Status'] = "Unclassified"

    # --- DROP ROWS marked for removal ---
    if rows_to_drop:
        print(f"\nDropping rows (indices): {rows_to_drop}")
        df.drop(index=rows_to_drop, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # ---- OUTPUT FORMATTING SECTION ----
    # 1. Rename Interest Rate column
    if 'Interest Rate' in df.columns:
        df = df.rename(columns={'Interest Rate': 'Interest/Dividend'})

    # 2. Ensure all value columns are NUMERIC (not string with $ or %)
    for col in ['Price', 'Market Value', 'Cost Basis', 'Gain/Loss $']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace('[\$,]', '', regex=True), errors='coerce')

    # 3. Add Monthly Income (as number)
    if 'Monthly Income' not in df.columns and 'Interest/Dividend' in df.columns and 'Market Value' in df.columns:
        idx = list(df.columns).index('Interest/Dividend')+1
        def calc_monthly_income(row):
            try:
                mv = float(row.get('Market Value', 0) or 0)
                # Monthly income is always based on the current market value * interest/dividend rate
                rate_str = str(row.get('Interest/Dividend', '')).replace('%','')
                # If rate_str is a percent (e.g. 1.27), divide by 100
                rate = float(rate_str)/100 if rate_str not in ['', None] else 0.0
                return round(mv * rate / 12, 2) if rate > 0 else 0.0
            except Exception:
                return 0.0
        df.insert(idx, 'Monthly Income', df.apply(calc_monthly_income, axis=1))

    # 4. Format Interest/Dividend as percent string
    if 'Interest/Dividend' in df.columns:
        def format_percent_cell(val):
            try:
                if isinstance(val, str) and '%' in val:
                    return val
                if val not in [None, '', 0, 0.0] and not pd.isna(val):
                    val = float(val)
                    return f"{val:.2f}%"
                else:
                    return ''
            except Exception:
                return ''
        df['Interest/Dividend'] = df['Interest/Dividend'].apply(format_percent_cell)

    return df








