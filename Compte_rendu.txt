
Mon ordinateur est dans la configuration : 
    CPU family:          6
    Model:               126
    Thread(s) per core:  2
    Core(s) per socket:  4
    Socket(s):           1
    Stepping:            5
    BogoMIPS:            2995.20

Caches (sum of all):     
  L1d:                   192 KiB (4 instances)
  L1i:                   128 KiB (4 instances)
  L2:                    2 MiB (4 instances)
  L3:                    8 MiB (1 instance)

Le problème est que les calculs se font tranche par tranche. Or, l'algorithme est basé 
sur l'utilisation des valeurs voisines. Comme aucune communication n'a été créée pour partager
les valeurs de bord, les tranches agissent indépendemment les unes des autres. C'est donc pour cette raison 
qu'il peut y avoir des problèmes de parties non colorées. 
On obtient un temps total d'exécution de 90.596 pour 1 processeur et de 56.112 pour 2 processeurs. On obtient un speedup de 1,615.

J'ai parallélisé les produits de matrices présents dans la fonction minimize.
Pour colorize2.py avec 1 processeur :

Temps application dirichlet sur matrice : 8.360506296157837 secondes
Temps calcul min Cb : 71.32961821556091
Temps calcul min Cr : 62.18721652030945

Pour colorize2.py avec 4 processeurs : 
Temps application dirichlet sur matrice : 16.693629264831543 secondes
Temps calcul min Cb : 146.733784198761
Temps calcul min Cr : 167.36252760887146

Le temps d'application dirichelet a augmenté alors que le code n'a pas été modifié ce qui me semble anormal.