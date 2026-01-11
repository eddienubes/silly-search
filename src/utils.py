from datetime import datetime


def get_readable_date() -> str:
    return datetime.now().strftime("%a %b %d, %Y")
