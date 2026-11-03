import pandas as pd
# print(pd.__version__)
#
# data = pd.read_csv(r'C:\Users\Admin\OneDrive\Desktop\for school\Architecture of computers\Projects\data\raw\IMDB Dataset.csv')
# print(data.shape)
# print(data.head())
#
# print(data.isna().sum())
# print(data.value_counts('sentiment'))
#
# print(data.duplicated().sum())
# print(data.duplicated(subset = ['review']).sum())
#
# print(data['review'].str.strip().eq('').sum())
#
# clean_data = data.drop_duplicates(subset = ['review']).copy()
# print(clean_data.shape)
#
# print(clean_data.value_counts('sentiment'))
# print(clean_data['review'].str.contains("<br />", regex = False).sum())
# clean_data['review'] = clean_data['review'].str.replace("<br />", ' ', regex = False)
# print(clean_data["review"].str.contains(r"<br\s*/?>", regex=True).sum())
# print(clean_data.duplicated(subset=["review"]).sum())
#
# clean_data['review'] = clean_data['review'].str.replace(r'\s+', ' ', regex = True).str.strip()
# len_of_rows = clean_data["review"].str.len()
# print(len_of_rows.describe())
#
#
# from pathlib import Path
#
# output_path = Path(__file__).resolve().parent / "data" / "processed" / "cleaned_IMDB_Dataset.csv"
# output_path.parent.mkdir(parents = True, exist_ok = True)
# clean_data.to_csv(output_path, index=False)
# saved_data = pd.read_csv(output_path)
# print(saved_data.shape)
#
# print(clean_data.reset_index(drop=True).equals(
#     saved_data.reset_index(drop = True)
#  )
# )

train = pd.read_parquet("data/raw/test-00000-of-00001.parquet")
test = pd.read_parquet("data/raw/test-00000-of-00001.parquet")

print(train.isna().sum())
print(train.value_counts('label' , dropna=False))

print(test.isna().sum())
print(test.value_counts('label' , dropna=False))