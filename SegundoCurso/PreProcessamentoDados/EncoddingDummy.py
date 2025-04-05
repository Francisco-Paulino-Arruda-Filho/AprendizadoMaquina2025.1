import pandas as pd

music_df = pd.read_csv('music.csv')
music_dummies = pd.get_dummies(music_df, columns=['genre'])
music_dummies = pd.concat([music_dummies, music_df], axis=1)
music_dummies = music_dummies.drop(columns=['genre'])
print(music_dummies.head())