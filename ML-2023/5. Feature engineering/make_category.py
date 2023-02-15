"""Створення нових ознак: 'Category' квартири - по відповідному евристичному алгоритму
"""

import pandas as pd
import numpy as np
from typing import List

HOME_DIR = '5. Feature engineering'



def make_appartment_category(data: pd.DataFrame, centrals: List[str] ) -> pd.DataFrame:
    """формує колонку `category` в `data` шляхом співпідставлення
    характеристик квартири
    """
    econom_pattern = {
        "rooms"         : lambda x: x <= 2,
        "level"         : lambda x: x <=2,
        "levels"        : lambda x: x <= 9,
        "year"          : lambda x: x <= 1970,
        "area_total"    : lambda x: x < 30,
        "area_kitchen"  : lambda x: x < 8,
        "area_comfort"  : lambda x: x < 10,
    }
    business_pattern = {
        "rooms"         : lambda x: x == 2 or x == 3,
        "level"         : lambda x: x > 2,
        "levels"        : lambda x: x >= 10,
        "year"          : lambda x: x >= 2010,
        "area_total"    : lambda x: x >= 30 and x <= 100,
        "area_kitchen"  : lambda x: x > 10,
        "area_comfort"  : lambda x: x > 10 and x < 20,
    }
    elit_pattern = {
        "rooms"         : lambda x: x >= 3,
        "area_total"    : lambda x: x >= 150,
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

    
def main():

    apartments: pd.DataFrame = pd.read_csv(HOME_DIR + '/apartments_final.csv')
    centrals:   pd.DataFrame = pd.read_csv(HOME_DIR + '/central.csv')
    
    apartments = make_appartment_category(apartments, list(centrals['name']))
    
    apartments.to_csv(HOME_DIR + "/apartments_category.csv", index=False)


if __name__ == '__main__':
    main()