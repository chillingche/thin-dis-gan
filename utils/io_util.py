import os


def make_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_filename_without_ext(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def get_lines_from_txt(txt):
    with open(txt, 'r') as f:
        lines = f.readlines()
    lines = [line.strip('\n').strip(' ') for line in lines]
    lines = filter(lambda x: x is not '', lines)
    return list(lines)


def write_lines_to_txt(lines, txt):
    lines = [line.strip(' ') for line in lines]
    lines = filter(lambda x: x is not '', lines)
    if lines is not None:
        lines = ["{}\n".format(line) for line in lines]
        with open(txt, 'w') as f:
            f.writelines(lines)
        print("Log is saved at:{}".format(txt))
    else:
        raise ValueError("Log is not a valid list of string.")
