import os
import datetime

def rename_to_creation_date(file_path: str = "trades.db") -> str:
    """
    Renames the given file to its creation date (YYYY-MM-DD.db).
    If a file with that name already exists, appends (1), (2), etc.
    Returns the new filename if successful, otherwise None.
    """
    if not os.path.exists(file_path):
        return None

    # Get creation time (Windows/macOS: real creation time, Linux: inode change time)
    creation_time = os.path.getctime(file_path)

    # Convert to YYYY-MM-DD
    date_str = datetime.datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d")

    # Base new filename
    base_name = f"{date_str}.db"
    new_name = base_name

    # Add suffix if file already exists
    counter = 1
    while os.path.exists(new_name):
        new_name = f"{date_str}({counter}).db"
        counter += 1

    # Rename
    os.rename(file_path, new_name)
    return new_name
