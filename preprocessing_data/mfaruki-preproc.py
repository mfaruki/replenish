import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
import string

# Importing the uncleaned dataframe
all_cuisine_df = pd.read_csv('all_cuisine_df.csv', index_col=0)

long_stopwords = ['score', 'teaspoon', 'tablespoons', 'chopped', 'fresh',
                'tablespoon', 'large', 'ground', 'teaspoons', 'finely',
                'ounces', 'sliced', 'kosher', 'plus', 'cut', 'inch',
                'divided', 'juice', 'unsalted', 'black', 'grated',
                'freshly', 'pound', 'red', 'peeled', 'thinly', 'ounce',
                'leaves', 'white', 'whole', 'pounds', 'extra',
                'vinegar', 'medium', 'water', 'small', 'virgin','purpose',
                'pieces', 'vegetable', 'coarsely', 'dried', 'seeds',
                'halved', 'minced', 'goes', 'with', 'baking', 'packed',
                'stick', 'serving', 'thick', 'slices', 'extract',
                'optional', 'dry', 'temperature', 'room', 'brown', 'trimmed',
                'sea', 'heavy', 'crushed', 'lengthwise', 'zest', 'toasted',
                'removed', 'cubes', 'preferably', 'diced', 'drained', 'fren',
                'chilled', 'broth', 'light', 'coarse',
                'pinch','sticks', 'taste', 'hot', 'unsweetened', 'flakes',
                'sweet', 'crosswise', 'stems', 'half','melted', 'dark',
                'granulated', 'low', 'quartered', 'pitted', 'flat', 'pan', 'torn', 'pure',
                'seeded', 'rinsed', 'baby', 'peel', 'italian', 'bunch', 'fine', 'yellow',
                'fat', 'paste', 'plain', 'head', 'ripe', 'bay', 'sodium', 'boneless',
                'sour', 'lightly', 'softened', 'strips', 'thawed', 'whipping',
                'crumbled', 'thin', 'bell', 'smoked','yolks', 'beaten', 'cooked',
                'dijon', 'ml', 'package', 'cored', 'piece', 'shredded', 'spray',
                'golden', 'skin', 'roasted', 'raw', 'bittersweet', 'tbsp', 'powdered',
                'total', 'fillets', 'additional', 'grams', 'cold',
                'ribs', 'long',  'style', 'separated', 'skinless',
                'flaky', 'bone', 'nonstick', 'greek', 'stalks',
                'mixed', 'canned', 'parts', 'drizzling',
                'see', 'wheat', 'tender', 'whites', 'frying', 'scrubbed', 'unpeeled',
                'bunches', 'split', 'one', 'firm', 'cooking', 'gold',
                'reduced', 'de', 'well', 'equipment', 'dusting', 'stemmed',
                'thermometer', 'deep', 'size', 'asian', 'rounds',
                'quality', 'inches', 'spice', 'shelled', 'accompaniment', 'semisweet',
                'allspoon', 'boiling', 'cracked', 'needed',
                'diagonal', 'supermarkets', 'soft', 'foods', 'stores', 'jar',
                'food', 'old', 'english', 'distilled',
                'homemade', 'quart', 'loosely', 'loaf', 'markets',
                'sprinkling', 'regular', 'hulled', 'center', 'sharp', 'unbleached',
                'assorted', 'sifted', 'dice', 'seedless', 'chops', 'free', 'scant',
                'short', 'spanish', 'seasoning',  'prepared',
                'use', 'ingredient', 'florets', 'pods', 'simple', 'blanched',
                'info', 'good', 'active', 'wild', 'packages', 'grill',
                'tart', 'shaved', 'parchment', 'french', 'cubed', 'hard',
                'tough', 'fresno', 'rings', 'bitters', 'sheets', 'minutes',
                'pressed', 'star', 'desired', 'roast', 'pink', 'mix', 'crust',
                'stalk', 'chinese', 'maldon', 'bite', 'dish', 'turkish', 'tea',
                'club', 'soaked', 'bottled', 'fronds', 'slightly', 'high', 'candy',
                'skinned', 'handful', 'flavored', 'california', 'seasoned', 'blade',
                'broken', 'ends', 'swiss', 'pints', 'two', 'dates',
                'greasing', 'mill', 'blue', 'new', 'romaine',
                'cured', 'tops', 'left', 'specialty', 'unflavored',
                'cleaned',  'smith', 'granny', 'boiled',
                'cooks', 'romano', 'recipe', 'bags', 'generous', 'quarts', 'fashioned',
                'sheet', 'surface', 'marjoram', 'persian', 'sparkling', 'bus',
                'end', 'chard', 'metal', 'box', 'diameter', 'slivered',
                'heirloom', 'purchased', 'mexican',
                'mini', 'bones', 'husked', 'picked', 'kitchen', 'fl',
                'preserves', 'skewers', 'shell', 'hearts',
                'dashes', 'container', 'serve', 'hothouse', 'chuck', 'nam',
                'ancho', 'pickled',
                'uncooked', 'slice', 'jack', 'including', 'called', 'also',
                'rolls', 'pestle', 'bottom', 'mortar', 'casings',
                'cutter',  'part', 'rind', 'wooden',
                'pans', 'muffin', 'árbol', 'threads','crystal',
                'matchsticks', 'intact', 'aleppo',
                'crusty', 'top', 'flaked', 'navel',
                'pearl', 'quick', 'julienned',
                'angostura', 'com', 'read', 'snap', 'blood',
                'made', 'mashed', 'meal', 'fry',  'graham',
                'crimini',  'coloring', 'outer',
                'slicer', 'diamond', 'dutch', 'like', 'creamy', 'attached',
                'spicy', 'meyer', 'rose', 'candied', 'lean',
                'firmly', 'fontina', 'sun', 'adjustable', 'shucked',
                'envelope', 'monterey', 'peeler', 'liquid', 'full',
                'warmed', 'eye', 'tip', 'butt', 'washed', 'korean', 'pepitas',
                'maker', 'aged', 'boston', 'processor', 'side', 'springform',
                'fitted', 'bowl', 'electric', 'dash',  'roasting',
                'four', 'mandoline', 'tuscan','crystallized',
                'twist', 'cheesecloth', 'neutral', 'grade', 'diagonally',
                'accompaniments', 'core', 'stem', 'brand', 'drops', 'using',
                'rising', 'cremini', 'plastic',  'lady',
                'sturdy', 'eow', 'round', 'qt', 'shallow', 'ceramic', 'glass',
                'storebought', 'reposado',  'grand', 'pat', 'depending',
                'wheels', 'salted', 'sweetness', 'amaro', 'marnier', 'averna', 'dehydrated',
                'wheel', 'amontillado','steamed', 'skirt', 'mixer', 'concentrate', 'rolled', 'containers',
                'poblano', 'glass', 'shaken', 'sometimes', 'grilled', 'found',
                'jumbo', 'scotch', 'escarole', 'horizontally',
                'self', 'gala', 'stone', 'rounded',
                'least', 'vegetables', 'removable', 'grinder',
                'skillet', 'string', 'strained', 'percent', 'sel',
                'pot', 'morton', 'fingerling',
                'wedge', 'knife', 'flank', 'grand', 'delicious',
                'process', 'marnier', 'spears', 'poppy', 'purple',
                'cane',  'matzo', 'skim', 'square', 'heaping',
                'fleur', 'racks', 'lengths', 'qt', 'necessary', 'choice',
                'semolina', 'sirloin', 'topping',
                'napa', 'fresco', 'ruby', 'rack', 'section', 'clean',
                'shaped','overnight',
                'cm', 'jars', 'pla',  'tips', 'hungarian', 'port', 'shavings',
                'little',  'molds', 'verts', 'grater', 'endive',
                'stand', 'excess',
                'lump', 'another',
                'favorite',  'per', 'phyllo', 'palm', 'wheel',
                'notes', 'stirred', 'cointreau', 'belgian', 'refrigerated',
                'puree', 'caps', 'littleneck', 'substitute',
                'glaze', 'table', 'bibb', 'bottles',
                'grilling', 'farro',  'day', 'curly', 'lady',
                'lowfat', 'bosc', 'five', 'spatula', 'baked',
                'crustless',  'iron', 'brewed', 'jelly', 'guajillo', 'strong',
                'smooth', 'weights', 'attachment', 'solid',
                'back', 'bits', 'wax', 'starch', 'range', 'fluted',
                'squares',  'rolling', 'sanding', 'labeled', 'combination',
                'slab', 'tails', 'foil', 'skins', 'links',
                'iceberg', 'lindt',
                'layers', 'tied',  'tube', 'leaving', 'latin',
                'juices', 'online', 'salata', 'less',
                'heart', 'thigh', 'grits', 'cutlets', 'crusts', 'live',
                'double', 'chervil', 'pop', 'shallow',
                'matchstick', 'measuring', 'hour', 'best', 'middle',
                'sold','savoy',
                'wheels', 'balls',  'depending', 'coating', 'ball',
                'young', 'breakfast', 'color', 'block',
                'leftover', 'fuji', 'anjou', 'seltzer', 'thickly',
                'demerara', 'plate', 'grind', 'better', 'cracker',
                'sized', 'cast', 'american', 'envelopes', 'rotisserie',
                'cube', 'indian', 'duty', 'slender', 'preserved', 'exceed',
                'castelvetrano', 'wrappers', 'special', 'quarters', 'blanc',
                'vegan', 'frenched', 'rabe', 'offset', 'cotija', 'edible', 'eastern',
                'heat', 'honeydew', 'acting', 'chanterelle', 'juniper', 'many', 'turbinado',
                'husks', 'still', 'lavender', 'snapper', 'mahi', 'beefsteak', 'malt', 'wafer',
                'kabocha', 'york', 'spelt', 'dressing', 'milliliter', 'fried', 'blanco',
                'caster', 'blackstrap', 'basic', 'butterflied', 'bartlett',
                'lima', 'kg', 'spiced', 'summer', 'crema', 'pinches', 'triple',
                'larger', 'evaporated', 'striped', 'marinated', 'challah', 'cooker',
                'fruity', 'without', 'debearded', 'batch',
                'semi', 'snow', 'grainy', 'aluminum', 'wrapped',
                'perugina', 'three', 'currant', 'julienne', 'layer', 'key',
                'splash', 'casing', 'twists', 'bonnet', 'thirds', 'flower',
                'rimmed', 'provolone', 'moons',
                'premium', 'st',  'calvados', 'sec', 'lukewarm',
                'blossoms', 'pod',  'oven',  'bella', 'fully',
                'marcona',   'mixture',
                'usually', 'flaxseed', 'disposable', 'jarred', 'tin',
                'holes', 'recipelink', 'andouille', 'bias', 'ribbons', 'ricer',
                'clear', 'colors', 'mission', 'delicata', 'winter', 'fleshed', 'imported',
                'king', 'tentacles', 'backbone',
                'san', 'scraped', 'meaty', 'pullman', 'eight',
                'whisked', 'wafers', 'spoon',
                'bulgur', 'according', 'stout', 'mexico', 'pounded',
                'segments', 'wood', 'unfiltered', 'shaoxing', 'drumstick', 'verde',
                'lid', 'sorbet', 'wiped', 'roll', 'pith', 'paddle',
                'flameproof', 'kirby', 'ear', 'flesh', 'several', 'bowls', 'dipping', 'tail', 'passion', 'tiny',
                'six', 'gem', 'boned', 'applewood',
                'variety', 'reserving', 'rustic', 'bodies', 'safflower',
                'colored', 'strings', 'shanks', 'bar', 'log', 'work',
               'amber', 'microplane', 'cassis', 'defrosted',
                'eyed', 'look', 'den', 'shank', 'la', 'blossom', 'sauvignon',
                'robust', 'loose',  'shoots', 'blender', 'make',
                'thaw', 'silken', 'fire', 'slow', 'icing', 'whisk',
                'ale', 'bitter', 'rainbow', 'gram',
                'juiced', 'sieve', 'press', 'masa', 'gouda', 'marzano',
                'sides', 'pin', 'bundt', 'big', 'need', 'honeycrisp',
                'cutters', 'rendered', 'walla', 'citrus', 'pernod', 'roots',
                'flavor',  'aperol', 'breads', 'manila', 'hemp', 'known',
                 'persimmons', 'decorating', 'maui', 'irish',
                'hours', 'added',  'standard', 'directions',
                'spread', 'brush', 'di',  'packaged', 'verbena',
                'malted', 'single', 'bonito', 'gum', 'charcoal', 'sorghum', 'herbes',
                'reserve', 'arctic', 'pickling', 'madras',
                'first', 'crispy', 'based', 'fluid', 'gray', 'pits',
                'treviso', 'dissolved', 'frosting', 'rub',
                'marked', 'niçoise', 'bottomed', 'add', 'chartreuse', 'pectin', 'pulp',
                'bird', 'dredging', 'stuffed',
                'guanciale', 'milliliters', 'mature', 'sift', 'endives', 'flax', 'find', 'farfalle',
                'forest', 'calabrian', 'cottage', 'pasilla', 'thickness', 'griddle',
                'fuyu', 'drumettes', 'heatproof',
                'franks', 'trans', 'puréed','together',
                'anaheim', 'farm', 'corned', 'stuffing',
                'proof', 'dust', 'resealable', 'buttering', 'individual',
                'ramps', 'holland', 'mesh',  'lillet',
                'capacity', 'classic', 'kind', 'tonic', 'steel', 'achiote',
                'minute', 'vera', 'grey',
                'planks', 'machine', 'measured', 'twine', 'basket', 'globe',
                'xanthan', 'mâche', 'wash', 'lebanese', 'cara',
                'sealable', 'deli', 'kieasa', 'mold', 'picholine', 'bakers',
                'buttered', 'enough', 'large', 'light', 'bag', 'of', 'pinch', 'mixture', 'ground', 'dried', 'from','piece',
                'boneless', 'cooked', 'steamed', 'easy-cook', 'boiled', 'minced', '-', 'or', 'and', 'sprig', 'weight', 'readycooked', 'x', 'tub', 'co', 'cup', 'just', 'a', 'over','tsp', 'each', 'in', 'pkt',
                'really', 'walnutsize', 'few', 'can', 'pointed', 'pack', 'about', 'packet', 'extramature', 'knob','rasher',
                'semidried','vegetarian' 'alternative', 'readytoeat', 'thumbsize', 'deseeded', 'crunchy', 'reserved', 'the', 'must', 'strip',
                'roughly', 'halffat','very','semiskimmed','family','carton','frozen','organic','peasized', 'blob','provence','stoned',
                'cob','nandouble','goodquality','readytobake','prime','ordinary', 'vegitarian','zeast','nanbone','nancarton','drizzle','autumn','fruit','we', 'used',
                'garden','mediumdry','sachet','quite','creamed','oz','pared','freerange','redskinned','bought','sashimigrade', 'flavoured','wrap', 'podded', 'finger',
                'alpro', 'drink','dribble','other','japanese', 'such','herb','eating',
                'freezedried','readytrimmed','sushigrade','for','quickcook','mild',
                'nanfrozen', 'readysliced','skinon','combined','sweetheart','thumbsized','vacuumpacked','chunk', 'heaped', 'cadburys', 'plenty', 'rindless', 'coldpressed', 'punnet', 'drop', 'fullfat',
                'jewelled', 'extralean', 'bottle', 'level', 'grating', 'precooked', 'skinon', 'bonein',
                'unroasted', 'ovenready', 'pint', 'drycured', 'peppered', 'freezedried', 'fastaction',
                'readymade', 'shellon', 'readysliced', 'but', 'greekstyle', 'dollop', 'pouch', 'mild', 'broad', 'mediumsized', 'readyroasted', 'unsmoked',
                'easybake', 'lot', 'johns', 'hotsmoked', 'british', 'feather', 'your', 'favourite', 'filling', 'unskinned', 'morello','squeeze','chunky','eg','fingerlength',
                'smallmedium','rizzazz','glutenfree','reducedfat','doublepodded', 'cloves of',
               ]

#Lemmatizer function
def lemmatizer(text):
    '''Row-wise Lemmatizer function that should be applied to a dataframe'''
    text = str(text)
    text = text.lower()
    text_lst = text.split(', ')

    lemmatizer = WordNetLemmatizer()
    lemmatized_output = [lemmatizer.lemmatize(w) for w in text_lst]

    return ' '.join(lemmatized_output)


def removed_stopwords(text):
    '''Row-wise function that removes the stopwords from the ingredients list, should be applied to a dataframe'''
    for punc in string.punctuation:
        text = text.replace(punc, '')

    text_lst = text.split()

    text_lst_no_nums = [t for t in text_lst if t.isalpha()]


    new_added = ['mediumsize']
    long_stopwords.extend(new_added)

    no_stopwords = []
    for t in text_lst_no_nums:
        if t not in long_stopwords:
            no_stopwords.append(t)


    # for sw in customized_stopwords:
    #     text = text.replace(sw, ' ')
    return no_stopwords



cleaned_df = all_cuisine_df.copy()
cleaned_df['lemmatized'] = cleaned_df['ingredients'].apply(lemmatizer)
cleaned_df['no_stop'] = cleaned_df['lemmatized'].apply(removed_stopwords)

cleaned_df.to_csv('cleaned.csv')
