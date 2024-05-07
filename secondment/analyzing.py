
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#----------results without context----------------------------
distances=["cosine","pearson","euclidean"]
initial_nonnegatives={
    1:{'cosine': {'precision': 0.6538461538461539, 'recall': 0.6538461538461539, 'mrr': 0.6538461538461539, 'mae': 0.10062767816378308, 'rmse': 0.19245319417509055}, 'euclidean': {'precision': 0.6596153846153846, 'recall': 0.6596153846153846, 'mrr': 0.6596153846153846, 'mae': 0.109399173935757, 'rmse': 0.1992890786505178}, 'pearson': {'precision': 0.6478632478632478, 'recall': 0.6478632478632478, 'mrr': 0.6478632478632478, 'mae': 0.10239738138195323, 'rmse': 0.20260884013801256}},
    2:{'cosine': {'precision': 0.5835897435897435, 'recall': 0.5835897435897435, 'mrr': 0.8782051282051281, 'mae': 0.11520006091769515, 'rmse': 0.21326263667068507}, 'euclidean': {'precision': 0.5580128205128205, 'recall': 0.5580128205128205, 'mrr': 0.8528846153846155, 'mae': 0.12164414014112887, 'rmse': 0.2163508762205774}, 'pearson': {'precision': 0.5745726495726496, 'recall': 0.5745726495726496, 'mrr': 0.8756410256410255, 'mae': 0.11297862728717128, 'rmse': 0.21644581127434043}},
    3:{'cosine': {'precision': 0.546153846153846, 'recall': 0.546153846153846, 'mrr': 0.925, 'mae': 0.10382337751040718, 'rmse': 0.19904992052706133}, 'euclidean': {'precision': 0.5239316239316238, 'recall': 0.5239316239316238, 'mrr': 0.8869658119658119, 'mae': 0.11054375151373688, 'rmse': 0.2024317107438595}, 'pearson': {'precision': 0.5498575498575499, 'recall': 0.5498575498575499, 'mrr': 0.9178774928774928, 'mae': 0.10395294173536172, 'rmse': 0.20632493434583857}}}
tot_results_initial={}
for k, d_distances in initial_nonnegatives.items():
    tot_results_initial.setdefault(k, {})
    for d in d_distances:
        tot_results_initial[k].setdefault(d, [])
        tot_results_initial[k][d].append(list(d_distances[d].values()))

print(tot_results_initial)


#-----------------results with context------------------------------
#current version  without  negatives  (run 20 times) 70-30%
context_nonegatives={1:[{'filesToProcess/ratings_t1-supervisor_False.csv': {'cosine': {'precision': 0.7243243243243244, 'recall': 0.7243243243243244, 'mrr': 0.7243243243243244, 'mae': 0.10975738292027688, 'rmse': 0.20726909142927324}, 'euclidean': {'precision': 0.7472972972972974, 'recall': 0.7472972972972974, 'mrr': 0.7472972972972974, 'mae': 0.11975915817855158, 'rmse': 0.2161626371921977}, 'pearson': {'precision': 0.7216216216216217, 'recall': 0.7216216216216217, 'mrr': 0.7216216216216217, 'mae': 0.11038334869746706, 'rmse': 0.2161330730788407}}}, {'filesToProcess/ratings_t1-doctoralstudent_False.csv': {'cosine': {'precision': 0.7083333333333334, 'recall': 0.7083333333333334, 'mrr': 0.7083333333333334, 'mae': 0.11109545392696556, 'rmse': 0.21162065936388266}, 'euclidean': {'precision': 0.7194444444444443, 'recall': 0.7194444444444443, 'mrr': 0.7194444444444443, 'mae': 0.11776350058950887, 'rmse': 0.21296523496830688}, 'pearson': {'precision': 0.7, 'recall': 0.7, 'mrr': 0.7, 'mae': 0.11065943375534797, 'rmse': 0.21487193069764168}}}, {'filesToProcess/ratings_t2-supervisor_False.csv': {'cosine': {'precision': 0.7000000000000001, 'recall': 0.7000000000000001, 'mrr': 0.7000000000000001, 'mae': 0.125628933342092, 'rmse': 0.22977089603537978}, 'euclidean': {'precision': 0.6838235294117647, 'recall': 0.6838235294117647, 'mrr': 0.6838235294117647, 'mae': 0.1293367112889246, 'rmse': 0.22686034394821272}, 'pearson': {'precision': 0.6705882352941177, 'recall': 0.6705882352941177, 'mrr': 0.6705882352941177, 'mae': 0.11917980723870106, 'rmse': 0.22947867238086195}}}, {'filesToProcess/ratings_t2-doctoralstudent_False.csv': {'cosine': {'precision': 0.6756756756756757, 'recall': 0.6756756756756757, 'mrr': 0.6756756756756757, 'mae': 0.11366615045318329, 'rmse': 0.2005213268380548}, 'euclidean': {'precision': 0.6594594594594594, 'recall': 0.6594594594594594, 'mrr': 0.6594594594594594, 'mae': 0.12200813563699249, 'rmse': 0.21306870186586338}, 'pearson': {'precision': 0.6630630630630633, 'recall': 0.6630630630630633, 'mrr': 0.6630630630630633, 'mae': 0.1117387879180708, 'rmse': 0.21205789662392543}}}],
2: [{'filesToProcess/ratings_t1-supervisor_False.csv': {'cosine': {'precision': 0.658108108108108, 'recall': 0.658108108108108, 'mrr': 0.9567567567567569, 'mae': 0.11191626711334472, 'rmse': 0.21203541887485886}, 'euclidean': {'precision': 0.6297297297297297, 'recall': 0.6297297297297297, 'mrr': 0.95, 'mae': 0.11378907819453443, 'rmse': 0.20516356592568089}, 'pearson': {'precision': 0.6378378378378379, 'recall': 0.6378378378378379, 'mrr': 0.9545045045045045, 'mae': 0.10700595524836211, 'rmse': 0.21085483396990307}}}, {'filesToProcess/ratings_t1-doctoralstudent_False.csv': {'cosine': {'precision': 0.6416666666666666, 'recall': 0.6416666666666666, 'mrr': 0.9597222222222221, 'mae': 0.10326467964403212, 'rmse': 0.19341619446235506}, 'euclidean': {'precision': 0.6159722222222223, 'recall': 0.6159722222222223, 'mrr': 0.9395833333333332, 'mae': 0.11164001846916985, 'rmse': 0.19937768727806965}, 'pearson': {'precision': 0.6282407407407408, 'recall': 0.6282407407407408, 'mrr': 0.9513888888888887, 'mae': 0.10570751296373795, 'rmse': 0.20508225050823647}}}, {'filesToProcess/ratings_t2-supervisor_False.csv': {'cosine': {'precision': 0.5897058823529411, 'recall': 0.5897058823529411, 'mrr': 0.9411764705882353, 'mae': 0.11474289593985702, 'rmse': 0.20648157857906702}, 'euclidean': {'precision': 0.5595588235294118, 'recall': 0.5595588235294118, 'mrr': 0.8985294117647058, 'mae': 0.12631276022868104, 'rmse': 0.2200550196160438}, 'pearson': {'precision': 0.5867647058823527, 'recall': 0.5867647058823527, 'mrr': 0.9245098039215685, 'mae': 0.11418803840731782, 'rmse': 0.2209926211449802}}}, {'filesToProcess/ratings_t2-doctoralstudent_False.csv': {'cosine': {'precision': 0.645945945945946, 'recall': 0.645945945945946, 'mrr': 0.9608108108108109, 'mae': 0.10593892696884417, 'rmse': 0.19940539363549967}, 'euclidean': {'precision': 0.6209459459459459, 'recall': 0.6209459459459459, 'mrr': 0.9317567567567568, 'mae': 0.10995955208763653, 'rmse': 0.1954517800919191}, 'pearson': {'precision': 0.6261261261261261, 'recall': 0.6261261261261261, 'mrr': 0.9454954954954954, 'mae': 0.10388104951324219, 'rmse': 0.20424708909001985}}}],
3:[{'filesToProcess/ratings_t1-supervisor_False.csv': {'cosine': {'precision': 0.5711711711711713, 'recall': 0.5711711711711713, 'mrr': 0.9603603603603604, 'mae': 0.10356439264383013, 'rmse': 0.2072997267685055}, 'euclidean': {'precision': 0.5518018018018018, 'recall': 0.5518018018018018, 'mrr': 0.9450450450450452, 'mae': 0.11483415292755587, 'rmse': 0.2119122147113624}, 'pearson': {'precision': 0.5675675675675674, 'recall': 0.5675675675675674, 'mrr': 0.9606606606606608, 'mae': 0.10815784294151999, 'rmse': 0.2149944681175861}}}, {'filesToProcess/ratings_t1-doctoralstudent_False.csv': {'cosine': {'precision': 0.5694444444444444, 'recall': 0.5694444444444444, 'mrr': 0.9481481481481481, 'mae': 0.11452081695907926, 'rmse': 0.2141758038463665}, 'euclidean': {'precision': 0.5560185185185187, 'recall': 0.5560185185185187, 'mrr': 0.9324074074074075, 'mae': 0.11833906710424902, 'rmse': 0.21208706693827425}, 'pearson': {'precision': 0.5663580246913581, 'recall': 0.5663580246913581, 'mrr': 0.9483024691358023, 'mae': 0.11308560634720836, 'rmse': 0.21750687906757632}}}, {'filesToProcess/ratings_t2-supervisor_False.csv': {'cosine': {'precision': 0.5245098039215687, 'recall': 0.5245098039215687, 'mrr': 0.9411764705882352, 'mae': 0.11565965958776066, 'rmse': 0.21073429460276488}, 'euclidean': {'precision': 0.5161764705882352, 'recall': 0.5161764705882352, 'mrr': 0.9132352941176471, 'mae': 0.1274183679798268, 'rmse': 0.22456283486241735}, 'pearson': {'precision': 0.5369281045751634, 'recall': 0.5369281045751634, 'mrr': 0.9393790849673203, 'mae': 0.1157163353176919, 'rmse': 0.22210207627915235}}}, {'filesToProcess/ratings_t2-doctoralstudent_False.csv': {'cosine': {'precision': 0.5702702702702703, 'recall': 0.5702702702702703, 'mrr': 0.9693693693693695, 'mae': 0.10546183420470204, 'rmse': 0.1937060214300363}, 'euclidean': {'precision': 0.5355855855855856, 'recall': 0.5355855855855856, 'mrr': 0.9466216216216216, 'mae': 0.11509472148210295, 'rmse': 0.20146697601095456}, 'pearson': {'precision': 0.5492492492492494, 'recall': 0.5492492492492494, 'mrr': 0.963063063063063, 'mae': 0.10756965917676674, 'rmse': 0.20679263205791942}}}]}

context_negatives={1:[{'filesToProcess/ratings_t1-supervisor_True.csv': {'cosine': {'precision': 0.7216216216216218, 'recall': 0.7216216216216218, 'mrr': 0.7216216216216218, 'mae': 0.1335932622018177, 'rmse': 0.23912114517192024}, 'euclidean': {'precision': 0.7337837837837837, 'recall': 0.7337837837837837, 'mrr': 0.7337837837837837, 'mae': 0.14134434097336712, 'rmse': 0.24352100139644714}, 'pearson': {'precision': 0.7135135135135136, 'recall': 0.7135135135135136, 'mrr': 0.7135135135135136, 'mae': 0.13549573645850016, 'rmse': 0.24410315292983903}}}, {'filesToProcess/ratings_t1-doctoralstudent_True.csv': {'cosine': {'precision': 0.7138888888888889, 'recall': 0.7138888888888889, 'mrr': 0.7138888888888889, 'mae': 0.13684522889588796, 'rmse': 0.2341547745443818}, 'euclidean': {'precision': 0.7138888888888888, 'recall': 0.7138888888888888, 'mrr': 0.7138888888888888, 'mae': 0.13848875198355795, 'rmse': 0.2334842159856178}, 'pearson': {'precision': 0.7, 'recall': 0.7, 'mrr': 0.7, 'mae': 0.13259530011344006, 'rmse': 0.23639856834916287}}}, {'filesToProcess/ratings_t2-supervisor_True.csv': {'cosine': {'precision': 0.723529411764706, 'recall': 0.723529411764706, 'mrr': 0.723529411764706, 'mae': 0.1468758906717739, 'rmse': 0.26237805120089075}, 'euclidean': {'precision': 0.6823529411764706, 'recall': 0.6823529411764706, 'mrr': 0.6823529411764706, 'mae': 0.1562994761007135, 'rmse': 0.27691109677785575}, 'pearson': {'precision': 0.673529411764706, 'recall': 0.673529411764706, 'mrr': 0.673529411764706, 'mae': 0.15273958907875437, 'rmse': 0.28051857927121093}}}, {'filesToProcess/ratings_t2-doctoralstudent_True.csv': {'cosine': {'precision': 0.6567567567567567, 'recall': 0.6567567567567567, 'mrr': 0.6567567567567567, 'mae': 0.14210512277280407, 'rmse': 0.2482398214568408}, 'euclidean': {'precision': 0.6675675675675674, 'recall': 0.6675675675675674, 'mrr': 0.6675675675675674, 'mae': 0.14591034342643788, 'rmse': 0.24819859136422728}, 'pearson': {'precision': 0.6675675675675676, 'recall': 0.6675675675675676, 'mrr': 0.6675675675675676, 'mae': 0.13953775871630042, 'rmse': 0.24954214013403087}}}],
                   2: [{'filesToProcess/ratings_t1-supervisor_True.csv': {'cosine': {'precision': 0.5945945945945945, 'recall': 0.5945945945945945, 'mrr': 0.8932432432432431, 'mae': 0.14351231296549702, 'rmse': 0.24858996604531014}, 'euclidean': {'precision': 0.5608108108108109, 'recall': 0.5608108108108109, 'mrr': 0.8763513513513512, 'mae': 0.14495330204680715, 'rmse': 0.24467366914132457}, 'pearson': {'precision': 0.5788288288288289, 'recall': 0.5788288288288289, 'mrr': 0.8941441441441441, 'mae': 0.13845758794103785, 'rmse': 0.24722143173751474}}}, {'filesToProcess/ratings_t1-doctoralstudent_True.csv': {'cosine': {'precision': 0.5902777777777779, 'recall': 0.5902777777777779, 'mrr': 0.8805555555555555, 'mae': 0.1354125074639475, 'rmse': 0.23947547622284118}, 'euclidean': {'precision': 0.5583333333333333, 'recall': 0.5583333333333333, 'mrr': 0.8597222222222223, 'mae': 0.14000438903724843, 'rmse': 0.23805186097194003}, 'pearson': {'precision': 0.5731481481481482, 'recall': 0.5731481481481482, 'mrr': 0.8759259259259259, 'mae': 0.13408959680674873, 'rmse': 0.24004286108516643}}}, {'filesToProcess/ratings_t2-supervisor_True.csv': {'cosine': {'precision': 0.5411764705882354, 'recall': 0.5411764705882354, 'mrr': 0.8470588235294118, 'mae': 0.15642311611418946, 'rmse': 0.2953186457110022}, 'euclidean': {'precision': 0.5110294117647058, 'recall': 0.5110294117647058, 'mrr': 0.8147058823529413, 'mae': 0.15693135204497446, 'rmse': 0.2834804327253178}, 'pearson': {'precision': 0.5161764705882353, 'recall': 0.5161764705882353, 'mrr': 0.8372549019607843, 'mae': 0.15390317723545707, 'rmse': 0.29104788339341153}}}, {'filesToProcess/ratings_t2-doctoralstudent_True.csv': {'cosine': {'precision': 0.5824324324324325, 'recall': 0.5824324324324325, 'mrr': 0.9067567567567568, 'mae': 0.13948489028528838, 'rmse': 0.23960322605334125}, 'euclidean': {'precision': 0.552027027027027, 'recall': 0.552027027027027, 'mrr': 0.8709459459459461, 'mae': 0.14276736728646544, 'rmse': 0.2411855246445617}, 'pearson': {'precision': 0.5621621621621621, 'recall': 0.5621621621621621, 'mrr': 0.8878378378378379, 'mae': 0.13990563208652892, 'rmse': 0.24935674923115492}}}],
                   3: [{'filesToProcess/ratings_t1-supervisor_True.csv': {'cosine': {'precision': 0.527027027027027, 'recall': 0.527027027027027, 'mrr': 0.9576576576576578, 'mae': 0.13682473511525503, 'rmse': 0.24779825538572814}, 'euclidean': {'precision': 0.5108108108108108, 'recall': 0.5108108108108108, 'mrr': 0.9416666666666667, 'mae': 0.13610148337878564, 'rmse': 0.24254554205956663}, 'pearson': {'precision': 0.5306306306306307, 'recall': 0.5306306306306307, 'mrr': 0.9584084084084084, 'mae': 0.13164903856892313, 'rmse': 0.24707007399662212}}}, {'filesToProcess/ratings_t1-doctoralstudent_True.csv': {'cosine': {'precision': 0.5564814814814814, 'recall': 0.5564814814814814, 'mrr': 0.9472222222222222, 'mae': 0.12842172615425612, 'rmse': 0.23292464352160375}, 'euclidean': {'precision': 0.5333333333333333, 'recall': 0.5333333333333333, 'mrr': 0.9284722222222224, 'mae': 0.1349382314982408, 'rmse': 0.23409799863167388}, 'pearson': {'precision': 0.5515432098765433, 'recall': 0.5515432098765433, 'mrr': 0.9478395061728394, 'mae': 0.12570214604944835, 'rmse': 0.23258564893083758}}}, {'filesToProcess/ratings_t2-supervisor_True.csv': {'cosine': {'precision': 0.5049019607843137, 'recall': 0.5049019607843137, 'mrr': 0.9333333333333332, 'mae': 0.1471034443545136, 'rmse': 0.2687602757825819}, 'euclidean': {'precision': 0.48529411764705876, 'recall': 0.48529411764705876, 'mrr': 0.9215686274509803, 'mae': 0.14486834904153653, 'rmse': 0.2625534609680026}, 'pearson': {'precision': 0.5045751633986927, 'recall': 0.5045751633986927, 'mrr': 0.9397058823529412, 'mae': 0.14015299337125514, 'rmse': 0.2646523781883884}}}, {'filesToProcess/ratings_t2-doctoralstudent_True.csv': {'cosine': {'precision': 0.5243243243243243, 'recall': 0.5243243243243243, 'mrr': 0.9653153153153153, 'mae': 0.13967858585170326, 'rmse': 0.24879451624199872}, 'euclidean': {'precision': 0.5058558558558559, 'recall': 0.5058558558558559, 'mrr': 0.9412162162162161, 'mae': 0.14129905779249757, 'rmse': 0.24707788848681264}, 'pearson': {'precision': 0.5246246246246247, 'recall': 0.5246246246246247, 'mrr': 0.9555555555555554, 'mae': 0.1335991220297314, 'rmse': 0.24444023221851127}}}]}

def dic_context_results(dic_context):
    tot_results_context={}
    for k, list_results in dic_context.items():
        tot_results_context[k]={}
        for results in list_results:
            for f, d_distances in results.items():
                for d in distances:
                    tot_results_context[k].setdefault(d,[])
                    tot_results_context[k][d].append(list(d_distances[d].values()))
    return tot_results_context

tot_results_context=dic_context_results(context_nonegatives)

print(tot_results_context)
def saving_results(tot_results):
    f=open("results.csv", "w")
    f.write("k,distance,value,precision,recall,mrr,mae,rmse\n")
    for k, d_distances in tot_results.items():
        print(k)
        for d, M in d_distances.items():
            print(d)
            M = np.array(M)
            f.write("{},{},mean,".format(k,d))
            f.write(",".join([str(x) for x in list(M.mean(axis=0))]) + "\n")
            f.write("{},{},std,".format(k, d))
            f.write(",".join([str(x) for x in list(M.std(axis=0))]) + "\n")
            print(M.mean(axis=0))

    f.close()



def barplotResults(filename):
    # Load the data into a pandas dataframe
    df = pd.read_csv(filename)
    df=df[df["value"]=="mean"]
    # Filter the dataframe to only include the "precision" column
    precision_df = df[["k","distance", "precision"]]

    # Plot the data using a grouped bar chart with values on top
    sns.set_style("whitegrid")
    sns.catplot(data=precision_df, x="k", y="precision", hue="distance", kind="bar",
                palette="muted", ci=None, alpha=.8, height=6)

    # Add the values on top of each bar
    ax = plt.gca()
    for p in ax.containers:
        ax.bar_label(p, label_type="edge", fontsize=10, padding=2)

    # Set the plot title and axis labels
    plt.title("Precision vs Distance for Different Values of K")
    plt.xlabel("K")
    plt.ylabel("Precision")

    plt.show()

def t (filename):
    df = pd.read_csv(filename)

saving_results(tot_results_context)
barplotResults("results.csv")