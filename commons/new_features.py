
import re
class learningRoute(object):
    ''''''
    def __init__(self):
        self.total_exp = None
        self.title_crystalcaves = [
            'Crystal Caves - Level 1',
            'Chow Time',
            'Balancing Act',
            'Chicken Balancer (Activity)', 
            'Lifting Heavy Things', 
            'Crystal Caves - Level 2',
            'Honey Cake',
            'Happy Camel',
            'Cart Balancer (Assessment)',
            'Leaf Leader',
            'Crystal Caves - Level 3',
            'Heavy, Heavier, Heaviest',
            'Pan Balance',
            'Egg Dropper (Activity)',
            'Chest Sorter (Assessment)'
        ]
        self.title_treetopcity = [
            'Tree Top City - Level 1',
            'Ordering Spheres',
            'All Star Sorting',
            'Costume Box',
            'Fireworks (Activity)',
            '12 Monkeys',
            'Tree Top City - Level 2',
            'Flower Waterer (Activity)',
            "Pirate's Tale",
            'Mushroom Sorter (Assessment)',
            'Air Show',
            'Treasure Map',
            'Tree Top City - Level 3',
            'Crystals Rule',
            'Rulers',
            'Bug Measurer (Activity)',
            'Bird Measurer (Assessment)'
        ]
        self.title_magmapeak = [
            'Magma Peak - Level 1',
            'Sandcastle Builder (Activity)',
            'Slop Problem',
            'Scrub-A-Dub',
            'Watering Hole (Activity)',
            'Magma Peak - Level 2',
            'Dino Drink',
            'Bubble Bath',
            'Bottle Filler (Activity)',
            'Dino Dive',
            'Cauldron Filler (Assessment)']

    def _set_dict(self):
        total_dic = dict()
        setattr(self, 'total_exp', total_dic)
        for k in ['title_crystalcaves', 'title_magmapeak', 'title_treetopcity']:
            k_dic = {title_:0 for title_ in getattr(self, k)}
            self.total_exp.update(k_dic.copy())
        return self
    def is_on_appropriate_route(self, world_text, Asessment_text):
        titlesInWorld = getattr(self, 'title_' + world_text.lower())
        _index = titlesInWorld.index(Asessment_text)
        ans = 1
        for title in titlesInWorld[0:_index]:
            ans *= self.total_exp.get(title)
        return ans
    def record_experiences(self, title_text, world_text):
        if world_text not in ['TREETOPCITY', 'CRYSTALCAVES', 'MAGMAPEAK']: return None
        self.total_exp.update({title_text:1})


class Title_levels(learningRoute):

    def __init__(self):
        super().__init__()
        self.titlesLevel ={}
        pass
    def set_titlesLevel(self):
        self._set_dict()
        titlesLevel ={}
        for k in ['title_crystalcaves', 'title_magmapeak', 'title_treetopcity']:
            world_title_list = getattr(self, k)
            level = 1
            dic_ = {}
            for title in world_title_list:
                res = re.findall(string=title, pattern=r'Level +(\d)')
                if res and res[0] != str(level):
                    level += 1
                dic_[title] = k+'_level_'+str(level)
            titlesLevel.update(dic_)
            setattr(self,'titlesLevel' , titlesLevel)
        pass
    def __getitem__(self, index):
        assert self.titlesLevel != {}, "You have to implement 'set_titlesLevel'method beforehand"
        return self.titlesLevel.get(index)

def countSession_eachlevels(count_dict, title_levels:Title_levels, title_text):
    # assert set(count_dict.keys()) hogehoge
    if title_levels[title_text]:
        count_dict[title_levels[title_text]] += 1
    return count_dict



if __name__ == "__main__":
    import pdb; pdb.set_trace()
    t_levels = Title_levels()
    t_levels.set_titlesLevel()
    print(t_levels.titlesLevel)
    # level別に分けられていることの確認
    # 簡単にunittest moduleって使えるんだっけ？
    assert t_levels['Ordering Spheres'] == 'title_treetopcity_level_1'
    assert t_levels['Dino Drink'] == 'title_magmapeak_level_2'
    assert t_levels['Flower Waterer (Activity)'] == 'title_treetopcity_level_2'