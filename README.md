Le projet se compose de quelques fichiers importants :
-learn.py : pour créer un model SVM et l'entrainer sur une dataset existante dans scikit-learn (load_breat_cancer), et ensuite décomposer les données en train et test pour vérifier le score du model et ensuite le sauvegarder dans le dossier courant.
-create_csv.py : pour créer un fichier .csv depuis la dataset sur sklearn, ce script est pour tester des fonctionnalités sur l'UI (User Interface).
-main.py : Le main programme où on trouve le code de notre application. L'application se décompose en trois parties:
		-La première partie: est designer à l'utilisateur ou il a le choix de faire des prédiction avec un formulaire ou bien charger un fichier csv.
		-La deuxième partie: Le formulaire: c'est une interface graphique ou l'utilisateur mets les données dans un formulaire a fin de faire une prédiction (cette interface est pratique pour prédire just un nombre limité de données), le formulaire contient un filtre à fin d'éviter les fautes de frappe (par exemple des lettres ou bien caractères spéciaux).
		-La troisième partie: Prédiction à partir d'un fichier csv, l'utilisateur charge un fichier csv et prédit plusieurs lignes à la fois.
