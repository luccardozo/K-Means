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

##################################
# Após a seleção do k ótimo visulamente
##################################
cl = pd.DataFrame()
"""
# Hobbies
cluster, it, wcss = kmeans(data['Hobbies'], 6)
for i in range(6):
    pd.DataFrame.from_dict(cluster[i]).to_csv('Hobbies' + str(i) + '.csv')


# Music and Movies
cluster, it, wcss = kmeans(data['Music'], 5)
for i in range(5):
    pd.DataFrame.from_dict(cluster[i]).to_csv('Music' + str(i) + '.csv')


# Person
cluster, it, wcss = kmeans(data['Person'], 6)
for i in range(6):
    pd.DataFrame.from_dict(cluster[i]).to_csv('Person' + str(i) + '.csv')


# Phobia
cluster, it, wcss = kmeans(data['Phobia'], 4)
for i in range(4):
    pd.DataFrame.from_dict(cluster[i]).to_csv('Phobia' + str(i) + '.csv')



# Socio
cluster, it, wcss = kmeans(data['Socio'], 3)
for i in range(3):
    pd.DataFrame.from_dict(cluster[i]).to_csv('Socio' + str(i) + '.csv')



# Spend
cluster, it, wcss = kmeans(data['Spend'], 4)
for i in range(4):
    pd.DataFrame.from_dict(cluster[i]).to_csv('Spend' + str(i) + '.csv')
"""