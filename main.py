from kmeans import *
import matplotlib.pyplot as plt

data = dict(Hobbies=pd.read_csv('data\HobbiesAndInterests_Vars.txt', sep='\t'),
            Music=pd.read_csv('data\MusicAndMovies_Vars.txt', sep='\t'),
            Person=pd.read_csv('data\Personality_Vars.txt', sep='\t'),
            Phobia=pd.read_csv('data\Phobias_Vars.txt', sep='\t'),
            Socio=cat_to_num(pd.read_csv('data\SocioDemographic_Vars.txt', sep='\t'),
                             ['Gender', 'Only.child', 'Education']),
            Spend=pd.read_csv('data\SpendingHabits_Vars.txt', sep='\t'))

wcss_array = {'Hobbies': np.array([]),
              'Music': np.array([]),
              'Person': np.array([]),
              'Phobia': np.array([]),
              'Socio': np.array([]),
              'Spend': np.array([])}


iter_array = {'Hobbies': np.array([]),
              'Music': np.array([]),
              'Person': np.array([]),
              'Phobia': np.array([]),
              'Socio': np.array([]),
              'Spend': np.array([])}

k_array = np.arange(1, 12, 1)

"""
# Teste do Metodo do Cotovelo
# Demora horas para completar as iterações de 1 até 12 clusters


for i in range(1, 12):
    for key, dset in data.items():
        cl, iter_a, wcss_a = kmeans(dset, i)
        iter_array[key] = np.append(iter_array[key], iter_a)
        wcss_array[key] = np.append(wcss_array[key], wcss_a)


for key, array in wcss_array.items():
    plt.plot(k_array, wcss_array[key])
    plt.xlabel('Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Dataset ' + key)
    plt.show()
    plt.plot(k_array, iter_array[key])
    plt.xlabel('Clusters')
    plt.ylabel('Number of Iterations')
    plt.title('Iterations for Dataset ' + key)
    plt.show()

"""