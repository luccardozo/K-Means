from kmeans import *
import matplotlib.pyplot as plt

data = {'hobbies': pd.read_csv('data\HobbiesAndInterests_Vars.txt', sep='\t'),
        'music': pd.read_csv('data\MusicAndMovies_Vars.txt', sep='\t'),
        'person': pd.read_csv('data\Personality_Vars.txt', sep='\t'),
        'phobia': pd.read_csv('data\Phobias_Vars.txt', sep='\t'),
        'socio': cat_to_num(pd.read_csv('data\SocioDemographic_Vars.txt', sep='\t'),
                            ["Gender", "Only.child", "Education"]),
        'spend': pd.read_csv('data\SpendingHabits_Vars.txt', sep='\t')}

"""
wcss_array = {'hobbies': np.array([]),
        'music': np.array([]),
        'person': np.array([]),
        'phobia': np.array([]),
        'socio': np.array([]),
        'spend': np.array([])} 
"""

wcss_array = np.array([])

for i in range(1, 12):
    """
    for key, dset in data.items():
        cl, c_iter[key], wcss_array[key] = kmeans(dset, i)
    """
    cl, c_iter, wcss = kmeans(data['hobbies'], i)  # DEMORA HORRORES
    wcss_array = np.append(wcss_array, wcss)

k_array = np.arange(1, 12, 1)
plt.plot(k_array, wcss_array)
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()
