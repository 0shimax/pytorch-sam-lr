import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_data(args:dict) -> pd.DataFrame:
    df = pd.read_csv(args["path"]).sort_values(args["sort_key"])
    df = df[df[args["is_app"]]]
    df = df[df.ssp_id!=5335]
    fetures, encoder = transform(df, args)
    return fetures, df[args["target"]].values, encoder


def transform(data:pd.DataFrame, args:dict) -> pd.DataFrame:
    data = data[args["select_cols"]]
    data[args["category_cols"]] = data[args["category_cols"]].astype(str)
    data = data.drop(args["drop_cols"] + [args["target"]], axis=1)

    if args["encoder"] is None:
        enc = OneHotEncoder(handle_unknown='ignore')
        encoder = enc.fit(data)
    else:
        encoder = args["encoder"]
    fetures = encoder.transform(data) # .toarray()
    return fetures, encoder    