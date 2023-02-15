"""unit-тести для програми 'feature_engineering'
"""
from feature_engineering import clear_data, create_district_column, \
    make_appartment_category, make_area_comfort_column, HOME_DIR
import pandas as pd
import numpy as np
from pytest import fixture

@fixture
def get_test_data():
    return pd.DataFrame([
        [1, 13, 12, 2013, 28,  5,  10,  'Оболонський'],
        [2, 18, 26, 2010, 64,  14, 13,  'Оболонський'],
        [5, 10, 10, 1961, 400, 50, 100, 'Печерський']
        ],
        columns=['rooms', 'levels', 'level', 'year', 'area_total',
                 'area_kitchen', 'area_comfort', 'district'])

def test_clear_data():
    # Test case 1: check if rows with more eq 50% NaN values are removed
    data = pd.DataFrame({'col1': [1, np.nan, 3, 5], 'col2': [np.nan, 5, np.nan, 7]})
    data = clear_data(data, None, .5)
    assert len(data) == 4
    assert data.iloc[0]['col1'] == 1

    # Test case 1: check if rows with more eq 0% NaN values are removed
    data = pd.DataFrame({'col1': [1, np.nan, 3, 5], 'col2': [np.nan, 5, np.nan, 7]})
    data = clear_data(data, ['col1'])
    assert len(data) == 3
    assert data.iloc[0]['col1'] == 1

    # Test case 2: check if rows with NaN values in fields in the 'fmt' list are removed
    data = pd.DataFrame({'col1': [1, 2, 3, np.nan], 'col2': [np.nan, 5, 6, 7]})
    fmt = ['col1']
    data = clear_data(data, fmt)
    assert len(data) == 3
    assert data.iloc[0]['col1'] == 1
    assert data.iloc[1]['col1'] == 2

# тестуємо створення колонки з районами
def test_create_district_column():
    
    data = pd.DataFrame({'street': ['Петропавлівська',
                                    'Ревуцького',
                                    'Дніпровська',
                                    np.nan],
                         'col1': np.arange(4)})
    districts = pd.DataFrame({
            'street':['Петропавлівська',
                      'Дніпровська',
                      'Івана Мазепи',
                      'Дніпровська',
                      'Введенська' ],
            'district': ['Борщагівський',
                         'Дніпровський',
                         'Печерський',
                         'Троєщина',
                         np.nan]})
    
    data:pd.DataFrame = create_district_column(data, districts)
    assert len(data) == 4
    assert data.iloc[0]['district'] == 'Борщагівський'
    assert np.isnan(data.iloc[1]['district'])


# тестуємо створення колонки `area_comfort`
def test_make_area_comfort_column():
    data = pd.DataFrame({
        'area_total': [100, 200, 50],
        'area_living': [80, np.nan, 40],
        'area_kitchen':[10, 25, 30]
        })
    data = make_area_comfort_column(data)
    assert data.shape == (3,4)
    assert data.iloc[0]['area_comfort'] == 10
    assert np.isnan(data.iloc[1]['area_comfort'])
    assert data.iloc[2]['area_comfort'] == 0


# тестуємо створення колонки `category`
def test_make_appartment_category(get_test_data):
    center: pd.DataFrame = pd.read_csv(HOME_DIR + '/centrtal.csv')
    test_data_axis_1_len = get_test_data.shape[1]
    data = make_appartment_category(get_test_data, center['name'].to_list())
    assert data.shape[1] ==  test_data_axis_1_len + 1
    assert data.loc[0]['category'] == 'економ'
    assert data.loc[1]['category'] == 'бізнес'
    assert data.loc[2]['category'] == 'еліт'
