"""Ticker display-name mapping."""

from __future__ import annotations

# Map symbol (or security_id) to a human-friendly name.
_DISPLAY_NAMES: dict[str, str] = {
    "RELIANCE": "Reliance Industries",
    "2885": "Reliance Industries",
    "TCS": "Tata Consultancy",
    "11536": "Tata Consultancy",
    "INFY": "Infosys",
    "1594": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "1333": "HDFC Bank",
    "SBIN": "State Bank of India",
    "3045": "State Bank of India",
    "ICICIBANK": "ICICI Bank",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "BHARTIARTL": "Bharti Airtel",
    "ITC": "ITC Ltd",
    "LT": "Larsen & Toubro",
    "AXISBANK": "Axis Bank",
    "WIPRO": "Wipro",
    "TATAMOTORS": "Tata Motors",
    "MARUTI": "Maruti Suzuki",
    "SUNPHARMA": "Sun Pharma",
    "BAJFINANCE": "Bajaj Finance",
    "HCLTECH": "HCL Technologies",
    "ADANIENT": "Adani Enterprises",
    "TATASTEEL": "Tata Steel",
    "NTPC": "NTPC",
    "POWERGRID": "Power Grid Corp",
    "ONGC": "ONGC",
    "COALINDIA": "Coal India",
    "HINDALCO": "Hindalco",
    "JSWSTEEL": "JSW Steel",
}


def ticker_display_name(ticker: str) -> str:
    """Return a display name for *ticker*, falling back to the ticker itself."""
    return _DISPLAY_NAMES.get(ticker.upper().strip(), ticker.upper().strip())
