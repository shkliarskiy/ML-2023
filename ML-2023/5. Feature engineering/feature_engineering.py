"""Програма призначена для реалізиції процедур перетвореня, 
відновлення та конструювання ознак в файлі 'apartments.csv',
а саме:
    1. Очищеняя даних: видаленя 'поганих' рядків (багато NaN)
    2. Відновлення пропущених ознак
    3. Створення нових ознак:
        3.1. 'District' - район розташування квартири
        3.2. 'Category' квартири - по відповідному евристичному алгоритму
Повертає новий файл в `apartments_new.csv`
"""

import pandas as pd
import numpy as np
from typing import List

HOME_DIR = '5. Feature engineering'

def clear_data(data: pd.DataFrame, 
               importants: List[str] | None = None,
               max_nan: float | None = None) -> pd.DataFrame:
    """видаляє рядки в яких
         - кількість пропущених значень більша ніж `max_nan`
         - якщо в рядку є пусте значення зі списку важливих
    Args:
        data (pd.DataFrame): вхідний датафрейм
        important_fields: список важливих ознак або []

    Returns:
        очищений датафрейм
    """
    
    # нічого не робити бо не задані параметри
    if not max_nan and not importants:
        return data
    
    
    # видалити рядки що задовільняють критерію (1)
    if max_nan:
        # визначити кількість допустимих пропущених значень
        row_length = data.shape[1]
        thresh = row_length * (1 - max_nan)
        
        data.dropna(thresh=thresh, inplace=True)
        
    if importants:
        data.dropna(subset=importants, inplace=True)
    
    return data    

def fill_missings(data: pd.DataFrame) -> pd.DataFrame:
    """заповнює пропущені значення (Nan)
    """
    # заповнити пусті числові колонки медіаной даної колонки (2)
    data.fillna({
        'rooms': data['rooms'].median(),
        'price_per_m2':data['price_per_m2'].median(),
        'level':data['level'].median(),
        'levels':data['levels'].median(),
        'year':data['year'].median(),
        'area_total':data['area_total'].median(),
        'area_living':data['area_living'].median(),
        'area_kitchen':data['area_kitchen'].median(),    
    }, inplace=True)
    
    # заповнити категоріальні поля найближчим попереднім(2)
    data.fillna(method='bfill', inplace=True)
    
    return data

def create_district_column(data: pd.DataFrame, districts: pd.DataFrame) -> pd.DataFrame:
    """для кожного значення з колонки `street` шукає в датасеті
    `districts` відповідний район та формує нову колонку в `data`
    """
    def _find_district(street_name: str) -> str:
        """повертає першу знайдену назву району з 'districts` для вулиці або NaN"""
        result = districts.loc[districts['street'] == street_name, 'district'].values
        return result[0] if len(result) >= 1 else np.nan
    
    # створити нову колонку
    dis_col = data['street'].map(lambda x: _find_district(x))    
    data['district']  = dis_col
    
    return data

def make_appartment_category(data: pd.DataFrame, centrals: List[str] ) -> pd.DataFrame:
    """формує колонку `category` в `data` шляхом співпідставлення
    характеристик квартири
    """
    econom_pattern = {
        "rooms"         : lambda x: x <= 2,
        "level"         : lambda x: x == 1,
        "levels"        : lambda x: x <= 5,
        "year"          : lambda x: x <= 1970,
        "area_total"    : lambda x: x < 30,
        "area_kitchen"  : lambda x: x < 7,
        "area_comfort"  : lambda x: x < 10,
    }
    business_pattern = {
        "rooms"         : lambda x: x == 2 or x == 3,
        "level"         : lambda x: x > 1,
        "levels"        : lambda x: x > 10,
        "year"          : lambda x: x >= 2010,
        "area_total"    : lambda x: x >= 30 and x <= 100,
        "area_kitchen"  : lambda x: x > 10,
        "area_comfort"  : lambda x: x > 10 and x < 20,
    }
    elit_pattern = {
        "rooms"         : lambda x: x >= 3,
        "area_total"    : lambda x: x >= 200,
        # "area_comfort"  : lambda x: x > y * .3,
        "district"      : lambda x: x in centrals
    }

    def _get_category(line: pd.Series) -> str | None :
        """вираховує категорію квартири на основі заданих шаблонів:
        створє 3 списки з відповідность ознак і вибирає той, де 
        співпадінь з шаблоном найбільше"""
        category = None
        econom: List[bool]   = [econom_pattern[key](line[key]) for key in econom_pattern.keys()]
        busines: List[bool]  = [business_pattern[key](line[key]) for key in business_pattern.keys()]
        elit: List[bool]     = [elit_pattern[key](line[key]) for key in elit_pattern.keys()]
        
        categiries_rank = {'економ': sum(econom),
                           'бізнес': sum(busines),
                           'еліт'  : sum(elit)}
        category = max(categiries_rank, key=categiries_rank.get)
        
        return category
        
    # зробити пусту колонку під 'категорії'
    data['category'] = np.nan
    # зробити список категорій для кожного рядка
    categories: List[str] = [_get_category(x) for _, x in data.iterrows()]
    # оновити колонку 'категорій'
    data['category'] = categories
    
    return data

def make_area_comfort_column(data: pd.DataFrame) -> pd.DataFrame:
    """формує колонку `comfort_area в `data` по формулі:
    comfort_living_area = area_total - kitchen_area - living_area"""
    data['area_comfort'] = data['area_total'] - data['area_living'] - data['area_kitchen']
    data.loc[data['area_comfort'] < 0, 'area_comfort'] = 0
    return data 

def main():
    # get raw file into dataset
    apartments: pd.DataFrame = pd.read_csv(HOME_DIR + '/apartments.csv')
    
    # видалити незначущі колонки/показчики
    apartments.drop(columns='publish_date', inplace=True)
    
    # видалити 'погані' рядки
    apartments = clear_data(apartments, max_nan=.5)
    
    # заповнити пропущені значення
    apartments = fill_missings(apartments)
    
    # додати колонку 'район'
    districts = pd.read_csv(HOME_DIR + '/kiev_districts.csv')
    apartments = create_district_column(apartments, districts)
    
    # розрахувати та додати колонку 'комфортна зона'
    apartments = make_area_comfort_column(apartments)
    
    apartments = fill_missings(apartments)
    
    # створити новий оновлений файл
    apartments.to_csv(HOME_DIR + "/apartments_new.csv", index=False)


if __name__ == '__main__':
    main()
