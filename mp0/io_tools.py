"""IO tools for mp0.
"""


def read_data_from_file(filename):
    """
    Read txt data from file.
    Each row is in the format article_id\ttitle\tpositivity_score\n.
    Store this information in a python dictionary. Key: article_id(int),
    value: [title(str), score(float)].

    Args:
        filename(string): Location of the file to load.
    Returns:
        out_dict(dict): data loaded from file.
    """
    out_dict = {}
    file_ = open(filename)
    data = file_.readlines()
    for line in data:
        article_id, title, score = line.strip().split("\t")
        out_dict[int(article_id)] = [str(title), float(score.strip("\n"))]

    file_.close()

    return out_dict


def write_data_to_file(filename, data):
    """
    Writes data to file in the format article_id\ttitle\tpositivity_score\n.

    Args:
        filename(string): Location of the file to save.
        data(dict): data for writting to file.
    """
    with open(filename, "w") as output:
        for article, val in data.items():
            output.write(str(article) + "\t" + str(val[0]) + "\t" + str(val[1]) + "\n")
