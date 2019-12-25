

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

