from datetime import datetime


def str2bool(v):
    return v.lower() in {
        'yes',
        'true',
        't',
        '1',
    }


def current_str_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
