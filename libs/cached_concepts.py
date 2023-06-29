import utils.const as const

def get_cached_concept(dataset_name):
    if dataset_name == const.WATERBIRDS_NAME:
            z_reject = [['a bird with aquatic habitat', 'a bird with terrestrial habitat'], ['a bird with keratin feathers physiology', 'a bird with hydrophobic feathers physiology'], ['a bird with insects diet', 'a bird with fish diet'], ['a bird with longer wingspan flight', 'a bird with shorter wingspan flight'], ['a bird with coastal migration', 'a bird with inland migration'], ['a bird that lives in watery environments', 'a bird that lives on land.'], ['a bird has feathers made of the protein', "a bird's physiology with feathers that rep"], ['a bird that eats bugs.', 'a bird that eats mainly fish.'], ['a bird with wings that span farther when', 'a bird with a smaller wingspan can'], ['a bird that migrates along coastlines', 'a bird that migrates to different areas']]
            z_accept = [['a bird with webbed feet', 'a bird with talons feet'], ['a bird with waterproof feathers', 'a bird with non-waterproof feathers'], ['a bird with larger size', 'a bird with smaller size'], ['a bird with darker color', 'a bird with lighter color'], ['a bird with longer bill', 'a bird with shorter bill'], ['a bird with wide beaks', 'a bird with narrow beaks']]
    elif dataset_name == const.CELEBA_NAME:
        z_reject = [['a person with dark skin tone', 'a person with light skin tone'], ['a person with angular strong facial features', 'a person with soft round facial features'], ['a person with high perceived attractiveness', 'a person with low perceived attractiveness'], ['a person with serious personality traits', 'a person with loving personality traits'], ['a person with high intelligence', 'a person with low intelligence'], ['a person with high confidence level', 'a person with low confidence level'], ['a person with a deep complexion.', 'a person with a fair complexion.'], ['a person with sharp, prominent facial features', 'a person with a gentle, rounded face'], ['an individual with deep-seated character', 'a person who is caring and kind.'], ['a highly intelligent individual.', 'a person of limited mental capacity.'], ['a person who is sure of themselves.', 'a person who lacks self-assurance']]
        z_accept = [['a person with dark hair', 'a person with blond hair'],['a person with coarse hair texture', 'a person with smooth hair texture'], ['a person with lighter eye color', 'a person with darker eye color']]
    elif dataset_name == const.PACS_NAME:
        spurious = [['overly loyal'], ['aggressive'], ['messy'], ['judgmental'], ['expensive'], ['difficult to play']]
        differences = [['this object is loyal'], ['this object has long trunk'], ['this object has long neck'], ['this object has self-awareness'], ['this object is shelter'], ['this object has strings']]
        spurious = np.unique(np.array(spurious).flatten()).flatten()
        differences = np.unique(np.array(differences).flatten()).flatten()
        z_reject = []
        visited = set()
        for i, item1 in enumerate(spurious):
            for j, item2 in enumerate(spurious):
                if (i, j) in visited:
                    continue
                if item1==item2:
                    continue
                z_reject.append([f'this object is {item1}',f'this object is {item2}'])
                visited.add((i,j))
                visited.add((j,i))
        z_accept = []
        visited = set()
        for i, item1 in enumerate(differences):
            for j, item2 in enumerate(differences):
                if (i, j) in visited:
                    continue
                if item1==item2:
                    continue
                z_accept.append([f'{item1}',f'{item2}'])
                visited.add((i,j))
                visited.add((j,i))
        z_accept = np.array(z_accept)
        z_reject = np.array(z_reject)
    elif dataset_name == const.CXR_NAME:
        z_reject = [['distorted lung contour', 'normal lung contour'], ['normal lung volume', 'decreased lung volume'], ['increased lung opacity', 'normal lung opacity'], ['present mediastinal shift', 'absent mediastinal shift']]
        z_accept = [['consolidation opacity', 'airspace opacity'], ['increased size', 'normal size'], ['symmetrical shape', 'uneven shape'], ['smooth border', 'irregular border']]
    elif dataset_name == const.VLCS_NAME:
        differences = [['feathers', 'wings'], ['engine', 'wheels'], ['legs', 'seat'], ['tail', 'fur'], ['arms', 'legs']]
        spurious = [['has four wheels'], ['has feathers'], ['barks'], ['has legs'], ['flies']]
        spurious = np.unique(np.array(spurious).flatten()).flatten()
        differences = np.unique(np.array(differences).flatten()).flatten()
        z_reject = []
        visited = set()
        for i, item1 in enumerate(spurious):
            for j, item2 in enumerate(spurious):
                if (i, j) in visited:
                    continue
                if item1==item2:
                    continue
                z_reject.append([f'this object {item1}',f'this object {item2}'])
                visited.add((i,j))
                visited.add((j,i))
        z_accept = []
        visited = set()
        for i, item1 in enumerate(differences):
            for j, item2 in enumerate(differences):
                if (i, j) in visited:
                    continue
                if item1==item2:
                    continue
                z_accept.append([f'this object has {item1}',f'this object has {item2}'])
                visited.add((i,j))
                visited.add((j,i))
    return z_reject, z_accept