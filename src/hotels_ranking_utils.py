from sklearn.model_selection import train_test_split
from pandas import DataFrame, Series, read_csv, factorize, to_datetime


def prepare_data(data: DataFrame) -> tuple:
	"""
	.

	:param data:
	:return:
	"""
	data_grp = data.groupby(['snapshot_date', 'checkin_date', 'day_diff', 'hotel_name', 'week_day'])
	idx = data_grp['discount_diff'].transform(max) == data['discount_diff']
	max_score = data[idx].drop_duplicates(subset=['snapshot_date', 'checkin_date', 'day_diff', 'hotel_name', 'week_day', 'discount_diff'])
	data_features = max_score[['snapshot_date', 'checkin_date', 'day_diff', 'hotel_name', 'week_day']].copy()
	Y_labels = max_score['discount_code'].copy()
	max_score['discount_code'].value_counts()
	fact_hotal_name = factorize(data_features['hotel_name'])[0]
	fact_week_day = factorize(data_features['week_day'])[0]
	fact_checkin_date = factorize(data_features['checkin_date'])[0]
	fact_snapshot_date = factorize(data_features['snapshot_date'])[0]
	X_data = data_features.copy()
	X_data['hotel_name'] = fact_hotal_name
	X_data['week_day'] = fact_week_day
	X_data['checkin_date'] = fact_checkin_date
	X_data['snapshot_date'] = fact_snapshot_date

	data_split_config = {
		'test_size': .3, 
		'random_state': 100
	}

	return (X_data, Y_labels), train_test_split(X_data, Y_labels, **data_split_config)  

def load_csv(path_in: str, path_out: str=None, save: bool=False) -> DataFrame:
	"""
	.
	
	:param path_in:
	:param path_out[=None]:
	:param save[=False]:
	:return:
	"""
	csv_in = read_csv(path_in)
	stadard_columns = list()
	for col in csv_in.columns:
		stadard_columns.append(col.lower().strip().replace(' ', '_'))
	cols_map = dict()
	for col, new_col in zip(csv_in.columns, stadard_columns):
		cols_map[col] = new_col
	csv_in = csv_in.rename(columns=cols_map)
	csv_out = csv_in.copy()
	csv_out['day_diff'] = (to_datetime(csv_in['checkin_date']) - to_datetime(csv_in['snapshot_date'])).dt.days
	csv_out['week_day'] = to_datetime(csv_in['checkin_date']).dt.day_name()
	csv_out['discount_diff'] = csv_in['original_price'] - csv_in['discount_price']
	csv_out['discount_perc'] = 100. * (csv_out['discount_diff'] / csv_in['original_price'])
	if save and path_out is not None:
		csv_out.to_csv(path_out, index=False)
	return csv_out

def confusion_matrix(cm: object) -> tuple:
	"""
	.
	
	:param cm:
	:return:
	"""
	row, rowNumber, TP, FN, FP, TN = 0, 1, 0, 0, 0, 0
	TP_rate, FN_rate, FP_rate, TN_rate = 0, 0, 0, 0
	for i, curri in enumerate(cm):
		for j, _ in enumerate(curri):
			if row == i and row == j:
				TP += cm[i][j]
			if row == i and row != j:
				FN += cm[i][j]
			if row != i and row == j:
				FP += cm[i][j]
			if row != i and row != j:
				TN += cm[i][j]
	
	TP_rate = 0 if TP == 0 and FN == 0 else TP / (TP + FN)
	FN_rate = 0 if FN == 0 and TP == 0 else FN / (FN + TP)
	FP_rate = 0 if TN == 0 and FP == 0 else FP / (FP + TN)
	TN_rate = 0 if TN == 0 and FP == 0 else TN / (TN + FP)
	return TP_rate, FN_rate, FP_rate, TN_rate

def get_price_by_index(hotels: DataFrame, hotel_name: DataFrame, checkin_date: object, discount_code: int) -> float:
	"""
	.

	:param hotels:
	:param hotel_name:
	:param checkin_date:
	:param discount_code:
	:return:
	"""
	results = hotels.loc[(hotels['hotel_name_index'] == hotel_name) & (hotels['checkin_date_index'] == checkin_date) & (hotels['discount_code'] == discount_code)]
	return -1 if results.size == 0 else results['discount_price'].sort_values(ascending=True)[:1].values[0]

def normalize(maxi: float, mini: float, v: float) -> float:
	"""
	.

	:param maxi:
	:param mini:
	:param v:
	:return:
	"""
	new_min, new_max = 0, 100
	return (((v - mini)/(maxi - mini)) * (new_max - new_min)) + new_min