import pandas as pd
from datetime import datetime
import math



def digits(value):
    cnt = 0
    for i in value:
        if '0' <= i <= '9':
            cnt += 1
    return cnt

def have(value):
    value = value.strip()
    if len(value) == 0:
        return 0
    return 1

def entropy(value):
    value = value.strip()
    p = {}
    for i in value:
        if i not in p:
            p[i] = 0
        p[i] += 1
    for i in p:
        p[i] /= len(value)
    ans = 0
    for i in p:
        ans -= p[i] * math.log(p[i])
    return ans

def calc_age(created_at):
    collect_year = 2020 #dataset collected during
    if created_at is None:
        return 365 * 2

    created_at_str = created_at.to_string()
    created_at_str = created_at_str.strip()

    mode = '%a %b %d %H:%M:%S %z %Y'
    if created_at_str.find('L') != -1:
        created_time = datetime.fromtimestamp(int(created_at.replace('000L', '')))
    else:
        created_time = datetime.strptime(created_at, mode)
    collect_time = datetime.strptime('{} Dec 31'.format(collect_year), '%Y %b %d')
    created_time = created_time.replace(tzinfo=None)
    collect_time = collect_time.replace(tzinfo=None)
    difference = collect_time - created_time
    return difference.days
  
def get_count(row, feature_name):
    if row[feature_name] is None:
        return 0
    else:
        return int(row[feature_name])

def get_bool_as_int(row, feature_name):
    return int(row[feature_name] == 'True ')

def get_length(row, feature_name):
    return len(row[feature_name].strip())

def get_digits(row, feature_name):
    return digits(row[feature_name])

def get_have(row, feature_name):
    return have(row[feature_name])

def get_entropy(row, feature_name):
    return entropy(row[feature_name])

def get_features(data):
    features = pd.DataFrame()
    profile = data['profile']
    features['profile_image_present'] = profile.apply(lambda x: (int(x['profile_image_url'].find('default_profile_normal') == -1)))
    features['listed'] = profile.apply(get_count, feature_name='listed_count')
    features['followers'] = profile.apply(get_count, feature_name='followers_count')
    features['tweets'] = profile.apply(get_count, feature_name='statuses_count')
    features['friends'] = profile.apply(get_count, feature_name='friends_count')
    features['favourites'] = profile.apply(get_count, feature_name='favourites_count') #TBR
    features['verified'] = profile.apply(get_bool_as_int, feature_name='verified') 
    features['screen_name_length'] = profile.apply(get_length, feature_name='screen_name')
    features['name_length'] = profile.apply(get_length, feature_name='name')
    features['screen_name_digits'] = profile.apply(get_digits, feature_name ='screen_name') #TBR
    features['name_digits'] = profile.apply(get_digits, feature_name ='screen_name')
    features['desc_length'] = profile.apply(get_length, feature_name='description') #TBR
    features['location_present'] = profile.apply(get_have, feature_name='location') 
    features['name_entropy'] = profile.apply(get_entropy, feature_name='name')
    features['screen_name_entropy'] = profile.apply(get_entropy, feature_name='screen_name')
    features['desc_entropy'] = profile.apply(get_entropy, feature_name='description')
    features['has_extended_profile'] = profile.apply(get_bool_as_int, feature_name='has_extended_profile') 
    features['default_profile'] = profile.apply(get_bool_as_int, feature_name='default_profile') 
    features['default_profile_image'] = profile.apply(get_bool_as_int, feature_name='default_profile_image') 
    features['label'] = data['label'].astype(int)

    return features

if __name__ == "__main__":
    data = pd.read_json('../data/all_data.json')
    data = get_features(data)
    data.to_csv("../data/features.csv", index=False)
