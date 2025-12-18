"""
Formatting utilities for the Streamlit UI.

Provides functions for formatting data, creating styled tables,
and generating download links.

Color Palette:
    Primary Color: #1f77b4 (Medium blue - buttons, links, accents)
    Background Color: #ffffff (White - main background)
    Secondary Background: #f0f2f6 (Light gray - cards, containers, sidebar)
    Text Color: #262730 (Dark charcoal - main text)
    Font: sans-serif
"""

import base64
import pandas as pd
import streamlit as st
from typing import Any, Union, List, Dict
from io import BytesIO

# Theme Color Constants
THEME_PRIMARY = "#1f77b4"
THEME_BACKGROUND = "#ffffff"
THEME_SECONDARY_BG = "#f0f2f6"
THEME_TEXT = "#262730"


def format_number(value: Union[int, float], decimals: int = 0) -> str:
    """
    Format a number with comma separators.

    Args:
        value: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted number string
    """
    if pd.isna(value):
        return "N/A"

    if decimals == 0:
        return f"{int(value):,}"
    else:
        return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a value as a percentage.

    Args:
        value: Value to format (0-100 or 0-1)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    if pd.isna(value):
        return "N/A"

    # Handle both 0-1 and 0-100 ranges
    if value <= 1.0:
        value = value * 100

    return f"{value:.{decimals}f}%"


def format_confidence_score(score: float) -> str:
    """
    Format a confidence score with visual indicator.

    Args:
        score: Confidence score (0-1)

    Returns:
        Formatted score with emoji indicator
    """
    if pd.isna(score):
        return "N/A"

    percentage = score * 100

    if score >= 0.8:
        emoji = "🟢"
    elif score >= 0.5:
        emoji = "🟡"
    else:
        emoji = "🔴"

    return f"{emoji} {percentage:.1f}%"


def highlight_confidence_scores(df: pd.DataFrame, column: str = 'Avg Confidence') -> pd.DataFrame:
    """
    Apply color styling to confidence scores in a dataframe.

    Args:
        df: DataFrame with confidence scores
        column: Name of the confidence score column

    Returns:
        Styled DataFrame
    """
    def color_confidence(val):
        if pd.isna(val):
            return ''

        if val >= 0.8:
            color = '#90EE90'  # Light green
        elif val >= 0.5:
            color = '#FFD700'  # Gold
        else:
            color = '#FFB6C1'  # Light pink

        return f'background-color: {color}'

    if column in df.columns:
        return df.style.applymap(color_confidence, subset=[column])
    else:
        return df.style


def style_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply professional styling to frequency tables.

    Args:
        df: Frequency table DataFrame

    Returns:
        Styled DataFrame
    """
    # Create a copy to avoid modifying original
    styled = df.style

    # Gradient on Count column
    if 'Count' in df.columns:
        styled = styled.background_gradient(
            subset=['Count'],
            cmap='Blues',
            vmin=0,
            vmax=df['Count'].max()
        )

    # Gradient on Percentage column
    if 'Percentage' in df.columns:
        styled = styled.background_gradient(
            subset=['Percentage'],
            cmap='Greens',
            vmin=0,
            vmax=df['Percentage'].max()
        )

    # Format numbers
    format_dict = {}
    if 'Count' in df.columns:
        format_dict['Count'] = '{:,.0f}'
    if 'Percentage' in df.columns:
        format_dict['Percentage'] = '{:.1f}%'
    if 'Avg Confidence' in df.columns:
        format_dict['Avg Confidence'] = '{:.3f}'

    if format_dict:
        styled = styled.format(format_dict)

    return styled


def format_dataframe_for_display(
    df: pd.DataFrame,
    max_col_width: int = 300,
    max_rows: int = None
) -> pd.DataFrame:
    """
    Format a DataFrame for better display in Streamlit.

    Args:
        df: DataFrame to format
        max_col_width: Maximum column width in pixels
        max_rows: Maximum number of rows to display

    Returns:
        Formatted DataFrame
    """
    display_df = df.copy()

    # Truncate long text columns
    for col in display_df.columns:
        if display_df[col].dtype == 'object':
            display_df[col] = display_df[col].apply(
                lambda x: str(x)[:100] + '...' if isinstance(x, str) and len(str(x)) > 100 else x
            )

    # Limit rows if specified
    if max_rows and len(display_df) > max_rows:
        display_df = display_df.head(max_rows)

    return display_df


def create_download_link(
    data: Union[pd.DataFrame, str, bytes],
    filename: str,
    link_text: str = "Download",
    file_format: str = 'csv'
) -> str:
    """
    Create a download link for data.

    Args:
        data: Data to download (DataFrame, string, or bytes)
        filename: Name of the downloaded file
        link_text: Text to display for the link
        file_format: Format of the file ('csv', 'excel', 'json', 'txt')

    Returns:
        HTML string for download link
    """
    if isinstance(data, pd.DataFrame):
        if file_format == 'csv':
            file_data = data.to_csv(index=False).encode()
            mime_type = 'text/csv'
        elif file_format == 'excel':
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                data.to_excel(writer, index=False)
            file_data = buffer.getvalue()
            mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif file_format == 'json':
            file_data = data.to_json(orient='records', indent=2).encode()
            mime_type = 'application/json'
        else:
            raise ValueError(f"Unsupported format for DataFrame: {file_format}")

    elif isinstance(data, str):
        file_data = data.encode()
        mime_type = 'text/plain'

    elif isinstance(data, bytes):
        file_data = data
        mime_type = 'application/octet-stream'

    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    b64 = base64.b64encode(file_data).decode()

    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'


def format_list_column(series: pd.Series, separator: str = ', ') -> pd.Series:
    """
    Format a Series containing lists into readable strings.

    Args:
        series: Pandas Series with list values
        separator: String to join list items

    Returns:
        Series with formatted strings
    """
    return series.apply(
        lambda x: separator.join(map(str, x)) if isinstance(x, list) else str(x)
    )


def create_metric_card(
    label: str,
    value: Any,
    delta: Any = None,
    help_text: str = None
) -> Dict[str, Any]:
    """
    Create a formatted metric card data structure.

    Args:
        label: Metric label
        value: Metric value
        delta: Change/delta value (optional)
        help_text: Tooltip help text (optional)

    Returns:
        Dictionary with metric data
    """
    return {
        'label': label,
        'value': value,
        'delta': delta,
        'help': help_text
    }


def format_code_list(codes: List[str], max_display: int = 3) -> str:
    """
    Format a list of codes for compact display.

    Args:
        codes: List of code IDs
        max_display: Maximum number of codes to show

    Returns:
        Formatted string
    """
    if not codes:
        return "None"

    if len(codes) <= max_display:
        return ', '.join(codes)
    else:
        visible = ', '.join(codes[:max_display])
        remaining = len(codes) - max_display
        return f"{visible} +{remaining} more"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_color_scale(value: float, min_val: float = 0, max_val: float = 1) -> str:
    """
    Generate a color based on a value in a range.

    Args:
        value: Value to map to color
        min_val: Minimum value in range
        max_val: Maximum value in range

    Returns:
        Hex color code
    """
    # Normalize value to 0-1
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)

    # Clip to 0-1
    normalized = max(0, min(1, normalized))

    # Map to color gradient (red -> yellow -> green)
    if normalized < 0.5:
        # Red to yellow
        r = 255
        g = int(255 * (normalized * 2))
        b = 0
    else:
        # Yellow to green
        r = int(255 * (1 - (normalized - 0.5) * 2))
        g = 255
        b = 0

    return f'#{r:02x}{g:02x}{b:02x}'


def format_table_header(columns: List[str]) -> str:
    """
    Format table headers for markdown tables.

    Args:
        columns: List of column names

    Returns:
        Markdown table header string
    """
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    return f"{header}\n{separator}"


def create_badge(text: str, color: str = "blue") -> str:
    """
    Create a colored badge for display.

    Args:
        text: Badge text
        color: Badge color ('blue', 'green', 'red', 'yellow', 'gray')

    Returns:
        HTML string for badge
    """
    # Color palette aligned with theme:
    # Primary: #1f77b4, Secondary BG: #f0f2f6, Text: #262730
    color_map = {
        'blue': '#1f77b4',      # Primary color
        'green': '#28a745',
        'red': '#dc3545',
        'yellow': '#ffc107',
        'gray': '#6c757d',
        'orange': '#fd7e14',
        'purple': '#6f42c1'
    }

    bg_color = color_map.get(color, color)

    return f'''
    <span style="
        background-color: {bg_color};
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 500;
        font-family: sans-serif;
    ">{text}</span>
    '''
