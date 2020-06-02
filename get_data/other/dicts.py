"""
Some dicts containing useful bits of information
"""
from definitions import *



# Get names
ucr = os.listdir(DATA_DIR + '/interim/ucr')
if os.path.exists(DATA_DIR + '/interim/uea'):
    uea = os.listdir(DATA_DIR + '/interim/uea')


# Some removals
remove = [
    'GesturePebbleZ1',
    'GesturePebbleZ2',
    'GestureMidAirD1',
    'DodgerLoopWeekend',
    'DodgerLoopGame',
    'DodgerLoopDay',
    'GestureMidAirD2',
    'ShakeGestureWiimoteZ',
    'MelbournePedestrian',
    'PickupGestureWiimoteZ',
    'AllGestureWiimoteX',
    'PLAID',
    'AllGestureWiimoteY',
    'GestureMidAirD3',
    'AllGestureWiimoteZ'
]

# These all have weird class distns
remove_2 = [
    'Fungi',
    'PigAirwayPressure',
    'PigCVP',
    'PigArtPressure'
]

remove += remove_2

for r in remove:
    if r in ucr:
        ucr.remove(r)


# Remove multivariate
remove = [
    # Have nans
    'CharacterTrajectories',
    'InsectWingbeat',
    'JapaneseVowels',
    'SpokenArabicDigits',

    # Huge dim
    'InsectWingbeat',
    'PEMS-SF',
    'MotorImagery',
    'Heartbeat',
    'FaceDetection',
    'DuckDuckGeese',

    # Min class size < 5
    # 'StandWalkJump',
    # 'ERing',
    # 'Handwriting'

    # Too long
    'EigenWorms'
]

uea_nans = [
    'CharacterTrajectories',
    # 'InsectWingbeat',
    'JapaneseVowels',
    'SpokenArabicDigits',
]
uea += uea_nans
#
# for r in remove:
#     if r in multivariate:
#         multivariate.remove(r)


ds_names = {
    'ucr': ucr,
    # 'uea_nans': uea_nans,
    'uea': uea,
}


