# useful tips for data competitions

- this repository will not show my solution until the competition is completed. 


What I learnt.
---
(Technical ones, which might show how stupid I am.)
1. groupby can be iterated.
2. Things can be looped like as follows 
  ```python
  for i, (j, group) in enumerate(df.groupby(xxx)):
   ```
   If you want to loop the groupby object, then you might be faced with tuple composed of (index,pd.DataFrame)
