import numpy as np
from inspect import signature
from collections import Counter # mnogo je brzi od list.count, vraca dict key = el, val = num_of_occurences
#====

# dodaj funkcije za distancu po volji ovde ispod

def EuclidianDistance(point1, point2):
    """
    Funkcija za Euklidovu distancu,
    d = sqrt suma (xi1 - xi2)**2

    input:  Rn vektor point1
            Rn vektor point2

    output: distanca izmedju point1 i point2
    """

    return (np.sum((point1 - point2)**2))**0.5


#====

def kNNClf(train, labels, target, k, dist_func, mode):
    """
    Funkcija za kNN klasifikaciju nacin rada opisan u info.txt

    input:  train tensor trening podataka, mora biti u obliku (broj podataka, duzina podatka)
            labels tensor labela trening podataka radi klasifikacije, mora biti u obliku (broj podataka, klasa*) *NE ONE HOT ENCODED VEC BAS BROJ
            target podatak koji treba da se klasifikuje
            k broj komsija koji ce se gledati
            dist_func funkcija po kojoj ce se racunati distanca target od train podatka, func mora da ima argumente u obliku (arg1, arg2)
            mode int moda rada

    output: ako mode 0 onda vraca P da je target neka klasa
            ako mode 1 onda vraca klasu sa najvecim P da je target
    """

    # assertovanje broja arg dist_funca
    assert len(signature(dist_func).parameters) == 2, "dist_func mora da ima samo dva argumenta"


    # racunanje svih distanci
    dist = list()
    for x in train:
        dist.append(dist_func(x, target))


    # nalazenje labela za minimalne distance
    neighbours = list()                         # lista za klase komsija targeta
    for i in range(k):
        idx = dist.index(min(dist))             # uzimam index najmanjeg u listi distanci da bih ga nasao u labels, pretpostavka naravno da su train i labels 1-1 preslikani
        neighbours.append(int(labels[idx]))     # list int jer Counter se ljuti na sve sto nije hashable (liste, array...)
        dist.remove(min(dist))                  # duplikati ne prave problem za remove u ovom slucaju jer min daje isto prvog na koji naidje


    # sredjivanje outputa po modu
    occurances = Counter(neighbours)        # dobijam dict pojava elemenata u neighbours
    classnum = len(np.unique(labels))       # ukupan broj klasa
    occurances = occurances.most_common()   # sada u occurances lista tuplea oblika (key, value)


    if mode == 0:
        P = np.zeros(shape = classnum)      # pravim arr nula velicina broja klasa i samo na mestima gde ima occurance te klase stavljam neku vrednost
        for pair in occurances:
            P[pair[0]] = 1.0 * pair[1] / k * 100  # vrednost koju stavljam je verovatnoca koliko je siguran da je to klasa
        
        return "Verovatnoce po klasama (%) {}".format(P)
    
    elif mode == 1:
        temp = np.array(occurances)
        idx = np.where(temp[:, 1] == temp[:, 1].max())  # nadji indeks najveceg broja u occurance
        
        return "Klasa {}, siguran {}%".format(temp[idx][0][0], 1.0*temp[idx][0][1] / k * 100) # vrati klasu koja je na tom indeksu